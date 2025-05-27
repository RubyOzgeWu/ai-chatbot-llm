""" Postman 單純 fastAPI """
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import os

""" 設置環境 """
# 載入 .env 檔案
load_dotenv()

# 讀取環境變數
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ 缺少 GOOGLE_API_KEY 環境變數，請確認 .env 設定！")

# 判斷是否在 Docker 容器內運行
RUNNING_IN_DOCKER = os.path.exists('/.dockerenv')

if RUNNING_IN_DOCKER:
    ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOSTS", "http://elasticsearch:9200")
else:
    ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOSTS_LOCAL", "http://localhost:9200")


# 確認環境變數是否正確載入
# print(f"🔍 GOOGLE_API_KEY: {GOOGLE_API_KEY}")
# print(f"🔍 ELASTICSEARCH_HOST: {ELASTICSEARCH_HOST}")

""" 連接 Elasticsearch """
es = Elasticsearch(ELASTICSEARCH_HOST)

""" 向量化模型 """
embedding_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")


# 測試連接
try:
    es.info()
    print("✅ Elasticsearch 連接成功！")
except Exception as e:
    print(f"❌ 無法連接到 Elasticsearch: {e}")

""" 導入 LLM 模型 """
genai.configure(api_key=GOOGLE_API_KEY)
model_name = "gemini-1.5-flash"  
llm = genai.GenerativeModel(model_name)


""" RAG 向量檢索方法 """
from numpy import dot
from numpy.linalg import norm

def cosine_score(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def normalize_article(article_text):
    import re
    CHINESE_NUMERAL_MAP = {
        "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
        "十": 10, "百": 100, "千": 1000
    }

    def chinese_to_int(text):
        if text.isdigit():
            return int(text)
        result, tmp = 0, 0
        units = {"十": 10, "百": 100, "千": 1000}
        for ch in text:
            if ch in units:
                if tmp == 0:
                    tmp = 1
                result += tmp * units[ch]
                tmp = 0
            else:
                tmp = CHINESE_NUMERAL_MAP.get(ch, 0)
        return result + tmp

    m = re.match(r"第([一二三四五六七八九十百零\d]+)條", article_text)
    if not m:
        return article_text
    raw = m.group(1)
    if raw.isdigit():
        return f"第{raw}條"
    return f"第{chinese_to_int(raw)}條"

def retrieve_similar_docs(query, index=["ai_immigration-law_index", "ai_immigration-regulations_index", "ai_nationality-law_index"], top_k=3, use_reference_expansion=True):
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
        if not es.indices.exists(index=index):
            print(f"❌ 錯誤：索引 '{index}' 不存在！請先建立索引並載入數據！")
            return []

        response = es.search(index=index, body=search_body)
        docs = [hit["_source"] for hit in response["hits"]["hits"]]

        if use_reference_expansion:
            extra_docs = []
            seen_keys = set((doc["name"], doc["number"]) for doc in docs)  # 避免重複
            for doc in docs:
                for ref in doc.get("reference_laws", []):
                    try:
                        normalized_article = normalize_article(ref["article"])
                        law_name = ref["law_name"]

                        ref_query = {
                            "query": {
                                "bool": {
                                    "must": [
                                        {"match": {"number": normalized_article}},
                                        {"match": {"name": law_name}}
                                    ]
                                }
                            }
                        }
                        ref_hits = es.search(index=index, body=ref_query)["hits"]["hits"]
                        for hit in ref_hits:
                            ref_doc = hit["_source"]
                            key = (ref_doc.get("name"), ref_doc.get("number"))
                            if key not in seen_keys:
                                extra_docs.append(ref_doc)
                                seen_keys.add(key)
                    except Exception as e:
                        print(f"⚠ 擴充查詢失敗: {e}")

            all_docs = docs + extra_docs
            all_docs = sorted(
                all_docs,
                key=lambda d: cosine_score(d["embedding"], query_embedding),
                reverse=True
            )
            docs = all_docs[:top_k]

        return [
            {
                "法規名稱": doc.get("name", ""),
                "修正日期": doc.get("date", ""),
                "章節標題": doc.get("chapter_title", ""),
                "條號": doc.get("number", ""),
                "內容": doc.get("content", "")
            }
            for doc in docs
        ]

    except Exception as e:
        print(f"❌ Elasticsearch 查詢失敗: {e}")
        return []

# def retrieve_similar_docs(query, index=["ai_immigration-law_index", "ai_immigration-regulations_index", "ai_nationality-law_index"], top_k=3):
#     query_embedding = embedding_model.encode(query).tolist()

#     search_body = {
#         "size": top_k,
#         "query": {
#             "script_score": {
#                 "query": {"match_all": {}},
#                 "script": {
#                     "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
#                     "params": {"query_vector": query_embedding}
#                 }
#             }
#         }
#     }

#     try:
#         if not es.indices.exists(index=index):
#             print(f"❌ 錯誤：索引 '{index}' 不存在！請先建立索引並載入數據！")
#             return []

#         response = es.search(index=index, body=search_body)

#         return [
#             {
#                 "法規名稱": hit["_source"].get("name", ""),
#                 "修正日期": hit["_source"].get("date", ""),
#                 "章節標題": hit["_source"].get("chapter_title", ""),
#                 "條號": hit["_source"].get("number", ""),
#                 "內容": hit["_source"].get("content", ""),
#             }
#             for hit in response["hits"]["hits"]
#         ]

#     except Exception as e:
#         print(f"❌ Elasticsearch 查詢失敗: {e}")
#         return []

""" RAG  檢索 + 生成回應 """   
def rag_fastapi(user_query):
    # RAG 檢索
    docs = retrieve_similar_docs(user_query)
    print(docs)

    if not docs:
        return  {
            "answer": "找不到相關條文。",
            "references": []
        }

    # RAG 搜尋 
    context = "\n\n".join([
        f"{doc['法規名稱']}（{doc['修正日期']}）\n{doc['章節標題']} {doc['條號']}\n{doc['內容']}"
        for doc in docs
    ])

    prompt = f"根據以下資訊回答問題:\n{context}\n\n問題: {user_query}"
    response = llm.generate_content(prompt)

    return {
      "answer": response.text
    }

