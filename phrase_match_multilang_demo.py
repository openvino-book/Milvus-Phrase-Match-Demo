from pymilvus import MilvusClient, DataType
import numpy as np

# ========================================
#   颜色辅助函数
# ========================================
def blue(t): return f"\033[94m{t}\033[0m"
def green(t): return f"\033[92m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def red(t): return f"\033[91m{t}\033[0m"
def bold(t): return f"\033[1m{t}\033[0m"
def cyan(t): return f"\033[96m{t}\033[0m"

URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COLLECTION_NAME = "multilang_phrase_demo_robust"

# ========================================
#   1. 建表
# ========================================
def setup_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    schema.add_field(
        field_name="text_en",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "english"},
    )

    # 坚持使用 chinese analyzer，证明我们在原生环境下也能跑通
    schema.add_field(
        field_name="text_zh",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "chinese"}, 
    )

    EMB_DIM = 8
    schema.add_field(field_name="embeddings", datatype=DataType.FLOAT_VECTOR, dim=EMB_DIM)

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embeddings", index_type="HNSW", metric_type="IP", params={"M": 8, "efConstruction": 64})
    client.create_index(COLLECTION_NAME, index_params)
    return EMB_DIM

# ========================================
#   2. 插入“防合并、防停用词”的测试数据
# ========================================
def insert_data(client: MilvusClient, emb_dim: int):

    english_data = [
        "The Milvus system is stable.",           # Slop 0
        "The Milvus vector system is stable.",    # Slop 1 (Insert 'vector')
        "The system Milvus is stable.",           # Slop 2 (Reverse)
        "The system called Milvus is stable."     # Slop 3 (Reverse + Insert 'called')
    ]

    chinese_data = [

        "今日北京上海航线正式开通", 
        
        "计划建设北京连接到上海的高速铁路", 

        "旅客往返于上海北京之间",         

        "这是一条由上海连接北京的干线"
    ]

    total = len(english_data)
    rng = np.random.default_rng(42)
    vectors = rng.random((total, emb_dim)).astype("float32")

    rows = []
    for en, zh, v in zip(english_data, chinese_data, vectors):
        rows.append({"text_en": en, "text_zh": zh, "embeddings": v.tolist()})

    client.insert(collection_name=COLLECTION_NAME, data=rows)
    client.load_collection(COLLECTION_NAME)
    print(f"已插入 {total} 条高鲁棒性测试数据。\n")

# ========================================
#   3. 查询逻辑
# ========================================
def phrase_query(client, field, target_phrase, slop_val):
    if slop_val is None:
        expr = f"PHRASE_MATCH({field}, '{target_phrase}')"
        label = "slop=0 (精确连续)"
    else:
        expr = f"PHRASE_MATCH({field}, '{target_phrase}', {slop_val})"
        label = f"slop={slop_val}"

    print(bold(cyan(f"▶ {expr}")))
    print(blue(f"  测试目标: {label}"))

    results = client.query(
        collection_name=COLLECTION_NAME,
        filter=expr,
        output_fields=["id", field]
    )

    count = len(results)
    print(f"  共命中: {count} 条")

    if count == 0:
        print(red("  (无匹配结果)"))
    else:
        results.sort(key=lambda x: x['id'])
        for r in results:
            text = r[field]
            # 简单的高亮逻辑（仅供显示）
            hl_terms = target_phrase.split()
            for t in hl_terms:
                text = text.replace(t, bold(red(t)))
            print(green(f"  ✓ [ID:{r['id']}] {text}"))

    print("-" * 60 + "\n")

# ========================================
#   Main Demo
# ========================================
def main():
    client = MilvusClient(uri=URI, token=TOKEN)
    emb_dim = setup_collection(client)
    insert_data(client, emb_dim)

    # ---------- 英文演示 ----------
    print(bold("========================================"))
    print(bold("🌍 英文 Phrase Match 演示"))
    print(bold("========================================"))

    phrase_query(client, "text_en", "Milvus system", None)
    phrase_query(client, "text_en", "Milvus system", 1)
    phrase_query(client, "text_en", "Milvus system", 2)
    phrase_query(client, "text_en", "Milvus system", 3)

    # ---------- 中文演示 ----------
    print(bold("========================================"))
    print(bold("🈺 中文 Phrase Match 演示"))
    print(bold("========================================"))

    phrase_query(client, "text_zh", "北京 上海", 1)
    phrase_query(client, "text_zh", "北京 上海", 3)
    phrase_query(client, "text_zh", "北京 上海", 5)
    phrase_query(client, "text_zh", "北京 上海", 7)

if __name__ == "__main__":
    main()