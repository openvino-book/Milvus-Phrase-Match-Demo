from pymilvus import MilvusClient, DataType
import numpy as np

# ========================================
#   ANSI Color Helper Functions
# ========================================
def blue(t): return f"\033[94m{t}\033[0m"
def green(t): return f"\033[92m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def red(t): return f"\033[91m{t}\033[0m"
def bold(t): return f"\033[1m{t}\033[0m"
def cyan(t): return f"\033[96m{t}\033[0m"


URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COLLECTION_NAME = "multilang_phrase_demo"


# ========================================
#   Create collection with multi-language analyzer
# ========================================
def setup_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True
    )

    # English analyzer
    schema.add_field(
        field_name="text_en",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "english"},
    )

    # Chinese analyzer（Jieba）
    schema.add_field(
        field_name="text_zh",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "chinese"},
    )

    EMB_DIM = 16
    schema.add_field(
        field_name="embeddings",
        datatype=DataType.FLOAT_VECTOR,
        dim=EMB_DIM
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embeddings",
        index_type="HNSW",
        metric_type="IP",
        params={"M": 8, "efConstruction": 64},
    )
    client.create_index(COLLECTION_NAME, index_params)

    return EMB_DIM


# ========================================
#   Insert multilingual data
#   （保留你现在的语料，只调整成小写英文 + 去掉句号，减少 analyzer 干扰）
# ========================================
def insert_data(client: MilvusClient, emb_dim: int):
    english = [
        "machine learning improves performance",
        "machine fast learning is widely used",
        "machine very fast learning yields better results",
        "learning machine methods are commonly adopted in research",
    ]

    chinese = [
        "向量 检索 性能 很 强",
        "向量 快速 检索 在 工程 中 很 常见",
        "向量 深度 复杂 检索 在 AI 中 很 关键",
        "检索 向量 方法 在 搜索 中 很 常见",
    ]

    total = len(english)
    rng = np.random.default_rng(42)
    vectors = rng.random((total, emb_dim)).astype("float32")

    rows = []
    for en, zh, v in zip(english, chinese, vectors):
        rows.append({"text_en": en, "text_zh": zh, "embeddings": v.tolist()})

    client.insert(collection_name=COLLECTION_NAME, data=rows)
    client.load_collection(COLLECTION_NAME)

    return english, chinese


# ========================================
#   Phrase Match With Color Output + Diff
# ========================================
def phrase(client, field, phrase, slop=None, prev_hits=None, title=""):
    if slop is None:
        expr = f"PHRASE_MATCH({field}, '{phrase}')"
    else:
        expr = f"PHRASE_MATCH({field}, '{phrase}', {slop})"

    print(bold(cyan(f"▶ {expr}")))

    results = client.query(
        collection_name=COLLECTION_NAME,
        filter=expr,
        output_fields=[field]
    )

    # 当前命中的文本集合（用来做差集）
    current = [r[field] for r in results]
    current_set = set(current)

    print(f"  共命中 {len(current)} 条")

    if not results:
        print(red("  (no matches)\n"))
        return current_set

    # 先把所有命中打印出来
    for t in current:
        print(green(f"  ✓ {t}"))

    # 如果有上一轮 slop 的结果，打印「相比上一次新增了什么」
    if prev_hits is not None:
        new_hits = current_set - prev_hits
        if new_hits:
            print(yellow("  ↑ 相比上一档 slop 新增匹配："))
            for t in new_hits:
                print(yellow(f"    + {t}"))
        else:
            print(yellow("  ↑ 相比上一档 slop 没有新增匹配（说明上一个 slop 已经足够覆盖所有变体）"))

    return current_set


# ========================================
#   Main Demo
# ========================================
def main():
    client = MilvusClient(uri=URI, token=TOKEN)
    emb_dim = setup_collection(client)
    insert_data(client, emb_dim)

    # ---------- 英文 ----------
    print(bold("============================"))
    print(bold("🌍 英文 Phrase Match 演示"))
    print(bold("============================"))

    prev = None

    print(blue("\n🔵 slop=0（必须连续）"))
    print("预期：只命中完全连续的 'machine learning'")
    prev = phrase(client, "text_en", "machine learning", None, prev)

    print(green("\n🟢 slop=1（允许插 1 个词）"))
    print("预期：允许 'machine X learning' 这类变体")
    prev = phrase(client, "text_en", "machine learning", 1, prev)

    print(yellow("\n🟡 slop=2（允许更多插词 & 轻微换序）"))
    print("预期：命中更多自然语言变体，包括部分词序变化")
    prev = phrase(client, "text_en", "machine learning", 2, prev)

    print(red("\n🔴 slop=3（继续放宽 slop）"))
    print("预期：如果还存在更极端的变体，这里会继续扩张；否则命中集合不再变化")
    prev = phrase(client, "text_en", "machine learning", 3, prev)


    # ---------- 中文 ----------
    print(bold("\n============================"))
    print(bold("🈺 中文 Phrase Match 演示"))
    print(bold("============================"))

    prev = None

    print(blue("\n🔵 slop=0（连续短语）"))
    print("预期：只命中连续出现『向量 检索』的句子")
    prev = phrase(client, "text_zh", "向量 检索", None, prev)

    print(green("\n🟢 slop=1（允许插入 1 个词）"))
    prev = phrase(client, "text_zh", "向量 检索", 1, prev)

    print(yellow("\n🟡 slop=2（允许更大间距）"))
    prev = phrase(client, "text_zh", "向量 检索", 2, prev)

    print(red("\n🔴 slop=3（允许更松的间距 / 词序变化）"))
    prev = phrase(client, "text_zh", "向量 检索", 3, prev)


if __name__ == "__main__":
    main()
