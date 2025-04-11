from elasticsearch import Elasticsearch, ConnectionError
from sentence_transformers import SentenceTransformer  # 向量化模型
from dotenv import load_dotenv
import os
import json
import uuid

load_dotenv()

ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")

""" 連接到 ES """
try:
    es = Elasticsearch(ES_HOST)

    # 測試連線
    if not es.ping():
        raise ConnectionError("❌ 無法連接到 Elasticsearch，請檢查伺服器是否正在運行。")

    print("✅ 成功連接到 Elasticsearch")
except ConnectionError as e:
    print(f"🔴 Elasticsearch 連線錯誤: {e}")
    exit(1)  # 退出程式

""" 初始化 embedding model """
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

""" 處理 labor-law.json 格式 """
# def handle_labor_law(json_data, index_name):
#     documents = json_data.get("章節", [])
#     if not documents:
#         print(f"⚠️ 檔案缺少 '章節'")
#         return 0, 0

#     if not es.indices.exists(index=index_name):
#         es.indices.create(index=index_name, body={
#             "mappings": {
#                 "properties": {
#                     "name": {"type": "keyword"},
#                     "date": {"type": "keyword"},
#                     "chapter_title": {"type": "keyword"},
#                     "number": {"type": "keyword"},
#                     "content": {"type": "text"},
#                     "embedding": {"type": "dense_vector", "dims": 512}
#                 }
#             }
#         })
#         print(f"✅ 建立 index：{index_name}")
#     else:
#         print(f"ℹ️ 索引已存在：{index_name}")

#     success, failed = 0, 0
#     for chapter in documents:
#         for clause in chapter.get("條款", []):
#             clause_id = clause.get("條號")
#             content = clause.get("內容", "")
            
#             if not clause_id or not content:
#                 failed += 1
#                 continue

#             embedding = model.encode(content).tolist()
#             doc_id = f"{chapter.get('章名', '')}_{clause_id}".replace(" ", "")

#             # 若已存在就跳過處理
#             if es.exists(index=index_name, id=doc_id):
#                 continue

#             try:
#                 es.index(index=index_name, id=doc_id, body={
#                     "name": json_data.get("法規名稱", ""),
#                     "date": json_data.get("修正日期", ""),
#                     "chapter_title": chapter.get("章節標題", ""),
#                     "number": clause_id,
#                     "content": content,
#                     "embedding": embedding
#                 })
#                 success += 1
#             except Exception as e:
#                 print(f"❌ 寫入失敗：{clause_id} - {e}")
#                 failed += 1

#     return success, failed

""" 處理 immigration-law.json 格式 """
def handle_immigration_law(json_data, index_name):
    documents = json_data.get("章節", [])
    if not documents:
        print(f"⚠️ 檔案缺少 '章節'")
        return 0, 0

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body={
            "mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "date": {"type": "keyword"},
                    "chapter_title": {"type": "keyword"},
                    "number": {"type": "keyword"},
                    "content": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": 512}
                }
            }
        })
        print(f"✅ 建立 index：{index_name}")
    else:
        print(f"ℹ️ 索引已存在：{index_name}")

    success, failed = 0, 0
    for chapter in documents:
        for clause in chapter.get("條文", []):
            clause_id = clause.get("條號")
            content = clause.get("內容", "")
            
            if not clause_id or not content:
                failed += 1
                continue

            embedding = model.encode(content).tolist()
            doc_id = f"{chapter.get('章名', '')}_{clause_id}".replace(" ", "")

            # 若已存在就跳過處理
            if es.exists(index=index_name, id=doc_id):
                continue

            try:
                es.index(index=index_name, id=doc_id, body={
                    "name": json_data.get("法規名稱", ""),
                    "date": json_data.get("修正日期", ""),
                    "chapter_title": chapter.get("章名", ""),
                    "number": clause_id,
                    "content": content,
                    "embedding": embedding
                })
                success += 1
            except Exception as e:
                print(f"❌ 寫入失敗：{clause_id} - {e}")
                failed += 1

    return success, failed


""" 在這裡增加其他處理格式 """


""" 資料來源路徑 """
DATA_DIR = "data"
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

""" 根據檔名選擇處理邏輯 """
for filename in files:
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"🔴 無法讀取 {filename}：{e}")
        continue

    # 自動用檔名轉 index 名
    base = os.path.splitext(filename)[0]  # e.g. "labor-law"
    index_name = f"ai_{base}_index"

    print(f"\n📄 處理檔案：{filename} → Index: {index_name}")

    # 根據檔案名稱決定格式處理方式
    # if "labor-law" in base:
    #     success, failed = handle_labor_law(json_data, index_name)
    if "immigration-law" in base:
        success, failed = handle_immigration_law(json_data, index_name)
    else:
        print(f"⚠️ 尚未支援此檔案格式：{filename}")
        continue

    print(f"✅ 寫入成功：{success} 筆，⚠️ 失敗：{failed} 筆")

print("\n📌 所有檔案處理完成！")




