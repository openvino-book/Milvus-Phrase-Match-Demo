from pymilvus import MilvusClient, DataType
import numpy as np


URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COLLECTION_NAME = "multilang_phrase_demo"


def setup_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)

    # 主键
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )

    # 英文字段（english analyzer）
    schema.add_field(
        field_name="text_en",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "english"},
    )

    # 中文字段（chinese analyzer）
    schema.add_field(
        field_name="text_zh",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "chinese"},  # 这里是 Jieba
    )

    # 向量字段
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

    # HNSW 索引
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

    print("Collection created.")
    return EMB_DIM


def insert_data(client: MilvusClient, emb_dim: int):
    print("Inserting data...")

    english_texts = [
        "Machine learning improves vector search performance.",
        "Vector search techniques are widely used in AI systems.",
        "Learning machine components optimize vector retrieval tasks.",
    ]

    chinese_texts = [
        "向量检索 技术 推动 了 AI 系统 的 发展。",
        "机器 学习 模型 与 向量 检索 的 结合 提升 了 性能。",
        "在 实际 应用 中，向量   深度   检索 方法 非常 重要。",
    ]

    rng = np.random.default_rng(seed=42)
    vectors = rng.random((len(english_texts) + len(chinese_texts), emb_dim)).astype("float32")

    data = []
    for i, (en, zh) in enumerate(zip(english_texts, chinese_texts)):
        data.append({
            "text_en": en,
            "text_zh": zh,
            "embeddings": vectors[i].tolist(),
        })

    client.insert(collection_name=COLLECTION_NAME, data=data)
    client.load_collection(COLLECTION_NAME)

    print("Data inserted and collection loaded.")
    return english_texts, chinese_texts, vectors



def run_phrase(client: MilvusClient, field: str, phrase: str, slop: int | None = None):
    if slop is None:
        f = f"PHRASE_MATCH({field}, '{phrase}')"
    else:
        f = f"PHRASE_MATCH({field}, '{phrase}', {slop})"

    print(f"\n=== {f} ===")
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter=f,
        output_fields=["id", field],
    )
    for r in results:
        print(f"[id={r['id']}] {r[field]}")


def main():
    client = MilvusClient(uri=URI, token=TOKEN)
    emb_dim = setup_collection(client)
    english_texts, chinese_texts, vectors = insert_data(client, emb_dim)

    print("\n\n====== 英文 Phrase Match ======")
    run_phrase(client, "text_en", "machine learning")
    run_phrase(client, "text_en", "machine learning", 1)
    run_phrase(client, "text_en", "machine learning", 2)

    print("\n\n====== 中文 Phrase Match ======")
    # 中文短语（连续）
    run_phrase(client, "text_zh", "向量 检索")
    run_phrase(client, "text_zh", "向量 检索", 1)
    run_phrase(client, "text_zh", "向量 检索", 2)

    # 倒序短语（模拟 slop=2）
    run_phrase(client, "text_zh", "检索 向量", 2)


if __name__ == "__main__":
    main()
