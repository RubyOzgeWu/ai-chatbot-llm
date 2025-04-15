# test.py

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# 讀取環境變數
load_dotenv()

# 判斷是否在 Docker 中
RUNNING_IN_DOCKER = os.path.exists('/.dockerenv')

if RUNNING_IN_DOCKER:
    ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOSTS", "http://elasticsearch:9200")
else:
    ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOSTS_LOCAL", "http://localhost:9200")

# 連接 Elasticsearch
es = Elasticsearch(ELASTICSEARCH_HOST)

# 向量模型
embedding_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# 檢索函式（與 main.py 中相同）
def retrieve_similar_docs(query, index=["ai_immigration-law_index", "ai_immigration-regulations_index", "ai_nationality-law_index"], top_k=3):
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
        return [hit["_source"] for hit in response["hits"]["hits"]]
    except Exception as e:
        print(f"❌ Elasticsearch 查詢失敗: {e}")
        return []


# 測試資料集
ground_truth = [
  {
    "query": "移民署什麼時候應該每年清理禁止出入國的案件？",
    "relevant_articles": ["第5條"]
  },
  {
    "query": "外國人被收容後，還可以怎麼處理？",
    "relevant_articles": ["第21條", "第24條"]
  },
]



def evaluate_precision_at_k(testset, top_k=3):
    hit_count = 0

    for item in testset:
        query = item["query"]
        relevant = item["relevant_articles"]
        results = retrieve_similar_docs(query, top_k=top_k)

        retrieved_ids = [r.get("number", "") for r in results]
        match = any(rel in rid for rel in relevant for rid in retrieved_ids)

        if match:
            hit_count += 1
        else:
            print(f"❌ 未命中：{query}")
            print(f"→ 期望: {relevant}")
            print(f"→ 實際: {retrieved_ids}")
            print("-" * 40)

    precision = hit_count / len(testset)
    print(f"\n📊 Precision@{top_k}: {precision:.2f}")


# CLI 執行入口
if __name__ == "__main__":
    evaluate_precision_at_k(ground_truth, top_k=3)
