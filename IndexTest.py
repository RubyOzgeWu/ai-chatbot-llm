from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import os
import re
from dotenv import load_dotenv

load_dotenv()

RUNNING_IN_DOCKER = os.path.exists('/.dockerenv')
ELASTICSEARCH_HOST = (
    os.getenv("ELASTICSEARCH_HOSTS", "http://elasticsearch:9200")
    if RUNNING_IN_DOCKER
    else os.getenv("ELASTICSEARCH_HOSTS_LOCAL", "http://localhost:9200")
)

es = Elasticsearch(ELASTICSEARCH_HOST)
embedding_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

CHINESE_NUMERAL_MAP = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000
}

def chinese_numeral_to_int(text):
    if text.isdigit():
        return int(text)

    result = 0
    tmp = 0
    units = {"十": 10, "百": 100, "千": 1000}
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in units:
            unit = units[ch]
            if tmp == 0:
                tmp = 1
            result += tmp * unit
            tmp = 0
        else:
            tmp = CHINESE_NUMERAL_MAP.get(ch, 0)
        i += 1
    result += tmp
    return result

def normalize_article(article_text):
    m = re.match(r"第([一二三四五六七八九十百零\d]+)條", article_text)
    if not m:
        return article_text

    raw = m.group(1)

    if raw.isdigit():  # 是阿拉伯數字，直接用
        return f"第{raw}條"

    # 是中文數字才轉換
    arabic = chinese_numeral_to_int(raw)
    return f"第{arabic}條"

def retrieve_similar_docs(query, index, top_k=3, use_reference_expansion=False):
    query_embedding = embedding_model.encode(query).tolist()
    search_body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
    }

    try:
        response = es.search(index=index, body=search_body)
        docs = [hit["_source"] for hit in response["hits"]["hits"]]
    except Exception as e:
        print(f"❌ Elasticsearch 查詢失敗: {e}")
        return []

    if use_reference_expansion:
        extra_docs = []
        for doc in docs:
            for ref in doc.get("reference_laws", []):
                try:
                    raw_article = ref["article"]
                    normalized_article = normalize_article(raw_article)
                    law_name = ref["law_name"]

                    print(f"\n📌 引用條文擴充中：")
                    print(f"🔹 law_name: {law_name}")
                    print(f"🔹 article (raw): {raw_article} → (normalized): {normalized_article}")

                    ref_query = {
                        "query": {
                            "bool": {
                                "must": [
                                    {"match": {"number": normalized_article}},
                                    {"match": {"name": law_name}},
                                    # {"term": {"name.keyword": law_name}}
                                ]
                            }
                        }
                    }

                    print(f"🧪 查詢條件: {ref_query}")
                    ref_index = index  # 用相同的 index 查
                    ref_res = es.search(index=ref_index, body=ref_query)
                    hits = ref_res["hits"]["hits"]
                    print(f"🔍 擴充查詢命中：{len(hits)} 筆")

                    for hit in hits:
                        print(f"✔ 命中條文：{hit['_source'].get('number')} - {hit['_source'].get('content', '')[:30]}...")
                        extra_docs.append(hit["_source"])

                except Exception as e:
                    print(f"⚠ 擴充條文查詢失敗: {e}")

        docs.extend(extra_docs)


    return docs

# -------- 測試資料 -------- #
ground_truth = [
    {
        "query": "未取得外國國籍是否可以撤銷國籍喪失？",
        "relevant_articles": ["第11條"]
    }
]

def evaluate_precision_at_k(testset, top_k=3, index="", label="", use_reference_expansion=False):
    hit_count = 0

    print(f"\n🚀 測試索引模式：{label}（index: {index}, 展開引用: {use_reference_expansion}）\n")
    for item in testset:
        query = item["query"]
        relevant = item["relevant_articles"]
        results = retrieve_similar_docs(query, index=index, top_k=top_k, use_reference_expansion=use_reference_expansion)
        retrieved_ids = [r.get("number", "") for r in results]

        match = any(rel == rid for rel in relevant for rid in retrieved_ids)
        if match:
            hit_count += 1
        else:
            print(f"❌ 未命中：{query}")
            print(f"→ 期望: {relevant}")
            print(f"→ 實際: {retrieved_ids}")
            print("-" * 40)

    precision = hit_count / len(testset)
    print(f"\n📊 Precision@{top_k}（{label}）: {precision:.2f}")

# -------- CLI 主入口 -------- #
if __name__ == "__main__":
    index = "ai_nationality-law_index"

    evaluate_precision_at_k(
        ground_truth,
        top_k=3,
        index=index,
        label="❌ 無引用條文擴充",
        use_reference_expansion=False
    )

    evaluate_precision_at_k(
        ground_truth,
        top_k=3,
        index=index,
        label="✅ 有引用條文擴充",
        use_reference_expansion=True
    )
