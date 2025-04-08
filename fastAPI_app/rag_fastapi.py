""" Postman 單純 fastAPI """
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import os

""" 設置環境 """
# 載入 .env 檔案
load_dotenv()

# 讀取環境變數
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ 缺少 OPENAI_API_KEY 环境变量，请确认 .env 设置！")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("❌ 缺少 GOOGLE_API_KEY 環境變數，請確認 .env 設定！")

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
client = OpenAI(api_key=OPENAI_API_KEY)
model_name = "gpt-3.5-turbo"
# genai.configure(api_key=GOOGLE_API_KEY)
# model_name = "gemini-1.5-pro"  
# llm = genai.GenerativeModel(model_name)


""" RAG 向量檢索方法 """
def retrieve_similar_docs(query, index="ai_labor-law_index", top_k=3):
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

        return [
            {
                "法規名稱": hit["_source"].get("name", ""),
                "修正日期": hit["_source"].get("date", ""),
                "章節標題": hit["_source"].get("chapter_title", ""),
                "條號": hit["_source"].get("number", ""),
                "內容": hit["_source"].get("content", ""),
            }
            for hit in response["hits"]["hits"]
        ]

    except Exception as e:
        print(f"❌ Elasticsearch 查詢失敗: {e}")
        return []

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

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个乐于助人的助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return {
            "answer": response.choices[0].message.content.strip()
        }

    except Exception as e:
        print(f"❌ OpenAI 回傳錯誤: {e}")
        return {
            "answer": "產生回應時發生錯誤。",
            "references": []
        }
    # response = openai.ChatCompletion.create(
    #     model=model_name,
    #     messages=[
    #         {"role": "system", "content": "你是一个乐于助人的助手。"},
    #         {"role": "user", "content": prompt}
    #     ]
    # )

    # return {
    #     "answer": response["choices"][0]["message"]["content"].strip()
    # }

    # response = llm.generate_content(prompt)

    # return {
    #   "answer": response.text
    # }

