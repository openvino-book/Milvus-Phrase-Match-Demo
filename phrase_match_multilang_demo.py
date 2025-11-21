from pymilvus import MilvusClient, DataType
import numpy as np

# ========================================
#  ANSI Color Helpers（终端高亮输出）
# ========================================
def blue(text): return f"\033[94m{text}\033[0m"
def green(text): return f"\033[92m{text}\033[0m"
def yellow(text): return f"\033[93m{text}\033[0m"
def red(text): return f"\033[91m{text}\033[0m"
def bold(text): return f"\033[1m{text}\033[0m"
def cyan(text): return f"\033[96m{text}\033[0m"


URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COLLECTION_NAME = "logs_phrase_demo"


# ========================================
#  Collection Setup
# ========================================
def setup_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )

    # 日志字段（英文日志，用 english analyzer）
    schema.add_field(
        field_name="log_text",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "english"},
    )

    EMB_DIM = 16
    schema.add_field(
        field_name="embeddings",
        datatype=DataType.FLOAT_VECTOR,
        dim=EMB_DIM,
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )

    # 索引
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embeddings",
        index_type="HNSW",
        metric_type="IP",
        params={"M": 8, "efConstruction": 64},
    )
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
    )

    return EMB_DIM


# ========================================
#  Insert log data
# ========================================
def insert_logs(client: MilvusClient, emb_dim):
    logs = [
        # 完整短语（标准错误）
        "error: connection reset by peer",

        # 插词版本
        "fatal: tcp connection reset by remote peer",

        # 多插词
        "connection was unexpectedly reset by the peer",

        # 倒序
        "peer reset the connection",

        # 错误误命中案例（BM25 会命中，但不是这个错误）
        "peer connection established successfully",

        # 内容相关但不是 reset
        "remote peer closed connection normally",

        # 完全不同的错误
        "connection timeout occurred",

        # 完全不同
        "peer authentication failed",
    ]

    rng = np.random.default_rng(seed=42)
    vectors = rng.random((len(logs), emb_dim)).astype("float32")

    data = [{"log_text": t, "embeddings": v.tolist()} for t, v in zip(logs, vectors)]
    client.insert(collection_name=COLLECTION_NAME, data=data)
    client.load_collection(COLLECTION_NAME)

    return logs, vectors


# ========================================
#  Phrase Match Query（彩色输出）
# ========================================
def phrase(client, phrase: str, slop: int | None = None):
    if slop is None:
        expr = f"PHRASE_MATCH(log_text, '{phrase}')"
    else:
        expr = f"PHRASE_MATCH(log_text, '{phrase}', {slop})"

    print(bold(cyan(f"▶ {expr}")))

    results = client.query(
        collection_name=COLLECTION_NAME,
        filter=expr,
        output_fields=["id", "log_text"],
    )

    if not results:
        print(red("  (no matches)\n"))
        return

    for r in results:
        print(green(f"  ✓ [id={r['id']}] {r['log_text']}"))


# ========================================
#  Baseline TEXT_MATCH（展示误命中）
# ========================================
def compare_baseline_TEXT_MATCH(client):
    print(bold(red("==============================")))
    print(bold(red("❌ BASELINE: TEXT_MATCH(log_text, 'connection AND peer')")))
    print(bold(red("（演示普通文本匹配的误命中问题）")))
    print(bold(red("==============================")))

    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="TEXT_MATCH(log_text, 'connection AND peer')",
        output_fields=["id", "log_text"],
    )

    for r in results:
        print(yellow(f"  ⚠ [id={r['id']}] {r['log_text']}"))


# ========================================
#  Main
# ========================================
def main():
    client = MilvusClient(uri=URI, token=TOKEN)
    emb_dim = setup_collection(client)
    logs, vectors = insert_logs(client, emb_dim)

    # Baseline：文本匹配误命中
    compare_baseline_TEXT_MATCH(client)

    # Phrase Match 演示
    print(bold("\n============================"))
    print(bold("🎯 Phrase Match 精确匹配演示"))
    print(bold("============================"))

    print(blue("🔵 slop=0：完全连续短语匹配"))
    phrase(client, "connection reset by peer")

    print(green("\n🟢 slop=1：允许插入 1 个词"))
    phrase(client, "connection reset by peer", 1)

    print(yellow("\n🟡 slop=2：允许多个插词"))
    phrase(client, "connection reset by peer", 2)

    print(red("\n🔴 slop=3：允许倒序 & 多插词"))
    phrase(client, "connection reset by peer", 3)


if __name__ == "__main__":
    main()
