from elasticsearch import Elasticsearch, ConnectionError
from sentence_transformers import SentenceTransformer  # 向量化模型
from dotenv import load_dotenv
import os
import json
import uuid
import re

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


# """ 索引優化：抽取參照法源 """
def extract_references(content, current_law):
    """
    回傳一個 list of dict，每個 dict 格式為：
    {
        "law_name": "國籍法",
        "article": "第3條",
        "paragraph": "第1項",
        "subparagraph": "第1款"
    }
    """
    if not content:
        return []

    results = []

    # 顯性引用
    explicit_block_pattern = re.compile(
        r"(?:依|根據|按|參照)[\s「『]*"
        r"(本法|[\u4e00-\u9fa5]{2,6}法)"
        r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
        r"((?:第[一二三四五六七八九十百\d]+項(?:、第[一二三四五六七八九十百\d]+項)*))?"
        r"((?:第[一二三四五六七八九十百\d]+款(?:、第[一二三四五六七八九十百\d]+款)*))?"
    )

    for match in explicit_block_pattern.finditer(content):
        law_name, article, item_block, subitem_block = match.groups()
        full_law = current_law if law_name == "本法" else law_name

        paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
        subparagraphs = re.findall(r"第[一二三四五六七八九十百\d]+款", subitem_block or "") or [None]

        for p in paragraphs:
            for s in subparagraphs:
                results.append({
                    "law_name": full_law,
                    "article": article,
                    "paragraph": p,
                    "subparagraph": s
                })

    # 隱性引用
    implicit_pattern = re.compile(
        r"(?<!法)(?<![\u4e00-\u9fa5])"
        r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
        r"((?:第[一二三四五六七八九十百\d]+項(?:、第[一二三四五六七八九十百\d]+項)*))?"
        r"((?:第[一二三四五六七八九十百\d]+款(?:、第[一二三四五六七八九十百\d]+款)*))?"
    )

    for match in implicit_pattern.finditer(content):
        article, item_block, subitem_block = match.groups()

        paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
        subparagraphs = re.findall(r"第[一二三四五六七八九十百\d]+款", subitem_block or "") or [None]

        for p in paragraphs:
            for s in subparagraphs:
                results.append({
                    "law_name": current_law,
                    "article": article,
                    "paragraph": p,
                    "subparagraph": s
                })

    return results

# def extract_references(content, current_law):
#     """
#     從條文內容中抽取參照法源，支援：
#     1. 顯性引用：如「依本法第十一條第二項、第三項」
#     2. 隱性引用：如「依第二十五條第三項」，預設為 current_law
#     3. 條文擴展：第十一條第二項、第三項 → 第11條第2項、第11條第3項
#     """
#     if not content:
#         return []

#     references = set()

#     # 顯性引用
#     explicit_block_pattern = re.compile(
#         r"(?:依|根據|按|參照)[\s「『]*"  # 前綴詞
#         r"(本法|[\u4e00-\u9fa5]{2,6}法)"  # 法名
#         r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"  # 條
#         r"((?:第[一二三四五六七八九十百\d]+項(?:、第[一二三四五六七八九十百\d]+項)*))?"  # 項
#         r"((?:第[一二三四五六七八九十百\d]+款(?:、第[一二三四五六七八九十百\d]+款)*))?"  # 款
#     )

#     for match in explicit_block_pattern.finditer(content):
#         law_name, article, item_block, subitem_block = match.groups()
#         full_law = current_law if law_name == "本法" else law_name
#         references.add(f"{full_law}-{article}")

#         if item_block:
#             for item in re.findall(r"第([一二三四五六七八九十百\d]+)項", item_block):
#                 references.add(f"{full_law}-{article}第{item}項")

#         if subitem_block:
#             for subitem in re.findall(r"第([一二三四五六七八九十百\d]+)款", subitem_block):
#                 references.add(f"{full_law}-{article}第{subitem}款")

#     # 隱性引用（如：第二十五條第三項）
#     implicit_pattern = re.compile(
#         r"(?<!法)(?<![\u4e00-\u9fa5])" 
#         r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
#         r"((?:第[一二三四五六七八九十百\d]+項(?:、第[一二三四五六七八九十百\d]+項)*))?"
#         r"((?:第[一二三四五六七八九十百\d]+款(?:、第[一二三四五六七八九十百\d]+款)*))?"
#     )

#     for match in implicit_pattern.finditer(content):
#         article, item_block, subitem_block = match.groups()
#         references.add(f"{current_law}-{article}")

#         if item_block:
#             for item in re.findall(r"第([一二三四五六七八九十百\d]+)項", item_block):
#                 references.add(f"{current_law}-{article}第{item}項")

#         if subitem_block:
#             for subitem in re.findall(r"第([一二三四五六七八九十百\d]+)款", subitem_block):
#                 references.add(f"{current_law}-{article}第{subitem}款")

#     return list(references)

""" 處理 immigration-law.json 格式 """
def handle_law(json_data, index_name):
    # 判斷是否為章節制
    is_chapter_based = "章節" in json_data

    # 根據結構取得條文清單
    if is_chapter_based:
        documents = json_data.get("章節", [])
    else:
        documents = [{"章名": None, "條文": json_data.get("條文", [])}]

    if not documents or all(not chapter.get("條文") for chapter in documents):
        print(f"⚠️ 檔案缺少條文資料")
        return 0, 0

    # 建立 index（若尚未存在）
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body={
            "mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "date": {"type": "keyword"},
                    "chapter_title": {"type": "keyword"},
                    "number": {"type": "keyword"},
                    "reference_laws": {
                        "type": "nested",
                        "properties": {
                            "law_name": {"type": "keyword"},
                            "article": {"type": "keyword"},
                            "paragraph": {"type": "keyword"},
                            "subparagraph": {"type": "keyword"}
                        }
                    },
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
        chapter_title = chapter.get("章名", "") or ""
        for clause in chapter.get("條文", []):
            clause_id = clause.get("條號")
            content = clause.get("內容", "")

            if not clause_id or not content:
                failed += 1
                continue

            embedding = model.encode(content).tolist()
            doc_id = f"{chapter_title}_{clause_id}".replace(" ", "")

            # 參照法源
            reference_laws = extract_references(content, json_data.get("法規名稱", ""))

            # 若已存在就跳過處理
            if es.exists(index=index_name, id=doc_id):
                continue

            try:
                es.index(index=index_name, id=doc_id, body={
                    "name": json_data.get("法規名稱", ""),
                    "date": json_data.get("修正日期", ""),
                    "chapter_title": chapter_title,
                    "number": clause_id,
                    "reference_laws": reference_laws, 
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
    if any(keyword in base for keyword in ["immigration-law", "immigration-regulations", "nationality-law"]):
        success, failed = handle_law(json_data, index_name)
    else:
        print(f"⚠️ 尚未支援此檔案格式：{filename}")
        continue

    print(f"✅ 寫入成功：{success} 筆，⚠️ 失敗：{failed} 筆")

print("\n📌 所有檔案處理完成！")




