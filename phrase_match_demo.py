from pymilvus import MilvusClient, DataType
import numpy as np

URI = "http://localhost:19530"
TOKEN = "root:Milvus"  # Milvus 默认用户名密码
COLLECTION_NAME = "tech_articles_phrase_demo"

def setup_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        print(f"Dropping old collection {COLLECTION_NAME} ...")
        client.drop_collection(COLLECTION_NAME)

    print(f"Creating collection {COLLECTION_NAME} ...")

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )

    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=1000,
        enable_analyzer=True,
        enable_match=True,
        # analyzer_params={"type": "english"},  # 英文为主的话可以打开
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

    print("Collection created and index built.")
    return EMB_DIM


def insert_demo_data(client: MilvusClient, emb_dim: int):
    print("Inserting demo data ...")

    texts = [
        "Machine learning boosts efficiency in large-scale data analysis.",
        "Machine deep learning models are widely adopted in production.",
        "Learning machine architectures optimize computational loads.",
        "Robotics applications of learning and machine control are booming.",
        "Vector databases power modern AI search systems.",
    ]

    rng = np.random.default_rng(seed=42)
    vectors = rng.random((len(texts), emb_dim)).astype("float32")

    data = [
        {"text": t, "embeddings": v.tolist()}
        for t, v in zip(texts, vectors)
    ]

    client.insert(
        collection_name=COLLECTION_NAME,
        data=data,
    )

    client.load_collection(COLLECTION_NAME)
    print("Inserted and loaded collection.")
    return texts, vectors


def run_phrase_query(client: MilvusClient, phrase: str, slop: int | None = None):
    if slop is None:
        filter_expr = f"PHRASE_MATCH(text, '{phrase}')"
        label = f"PHRASE_MATCH('{phrase}')"
    else:
        filter_expr = f"PHRASE_MATCH(text, '{phrase}', {slop})"
        label = f"PHRASE_MATCH('{phrase}', slop={slop})"

    print("\n=== " + label + " ===")
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter=filter_expr,
        output_fields=["id", "text"],
    )
    if not results:
        print("No match.")
        return
    for r in results:
        print(f"[id={r['id']}] {r['text']}")


def run_vector_search_with_phrase_filter(
    client: MilvusClient,
    query_vec: np.ndarray,
    phrase: str,
    slop: int | None = None,
):
    if slop is None:
        filter_expr = f"PHRASE_MATCH(text, '{phrase}')"
    else:
        filter_expr = f"PHRASE_MATCH(text, '{phrase}', {slop})"

    print(f"\n=== Vector search + {filter_expr} ===")
    results = client.search(
        collection_name=COLLECTION_NAME,
        anns_field="embeddings",
        data=[query_vec.tolist()],
        filter=filter_expr,
        search_params={"metric_type": "IP", "params": {"ef": 32}},
        limit=3,
        output_fields=["id", "text"],
    )

    for hit in results[0]:
        text = hit["entity"]["text"]
        score = hit["distance"]
        print(f"score={score:.4f}  text={text}")


def main():
    client = MilvusClient(uri=URI, token=TOKEN)
    print("Connected to Milvus.")

    emb_dim = setup_collection(client)
    texts, vectors = insert_demo_data(client, emb_dim)

    # 只看 Phrase Match 效果
    run_phrase_query(client, "machine learning")      # slop 默认 0
    run_phrase_query(client, "machine learning", 1)
    run_phrase_query(client, "machine learning", 2)

    # 向量检索 + Phrase Match 组合
    query_vec = vectors[0]
    run_vector_search_with_phrase_filter(
        client, query_vec, "machine learning", 2
    )


if __name__ == "__main__":
    main()

