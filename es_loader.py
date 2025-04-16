from elasticsearch import Elasticsearch, ConnectionError
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")

# 主法－細則對應表
LAW_RELATIONS = {
    "入出國及移民法施行細則": "入出國及移民法",
    "國籍法施行細則": "國籍法"
}

LAW_FAMILY = {
    "入出國及移民法": "入出國及移民系列",
    "入出國及移民法施行細則": "入出國及移民系列",
    "國籍法": "國籍法系列"
}

try:
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise ConnectionError("❌ 無法連接到 Elasticsearch。")
    print("✅ 成功連接到 Elasticsearch")
except ConnectionError as e:
    print(f"🔴 連線錯誤: {e}")
    exit(1)

model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

CHINESE_NUMERAL_MAP = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000
}

def chinese_numeral_to_int(text):
    num_map = CHINESE_NUMERAL_MAP
    result = 0
    tmp = 0
    last_unit = 1

    units = {"十": 10, "百": 100, "千": 1000}
    num = 0
    i = 0

    while i < len(text):
        ch = text[i]
        if ch in units:
            unit = units[ch]
            if tmp == 0:
                tmp = 1
            result += tmp * unit
            tmp = 0
            last_unit = unit
        else:
            tmp = num_map.get(ch, 0)
        i += 1

    result += tmp
    return result

def normalize_article(article_text):
    m = re.match(r"第([一二三四五六七八九十百零\d]+)條", article_text)
    if m:
        chinese_num = m.group(1)
        arabic = chinese_numeral_to_int(chinese_num)
        return f"第{arabic}條"
    return article_text

def extract_references(content, current_law):
    if not content:
        return []

    results = []

    alias_map = {
        "本法": LAW_RELATIONS.get(current_law, current_law),
        "本細則": current_law
    }

    explicit_patterns = [
        re.compile(
            r"(?:依|根據|按|參照)[\s「『]*"
            r"(本法|本細則|[\u4e00-\u9fa5]{2,6}法)"
            r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
            r"((?:第[一二三四五六七八九十百\d]+項)*)?"
            r"((?:第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)*)?"
        ),
        re.compile(
            r"(本法|本細則|[\u4e00-\u9fa5]{2,}法)"
            r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
            r"(第[一二三四五六七八九十百\d]+項)?"
            r"(第[一二三四五六七八九十百\d]+款)?"
            r"(所稱|所定|所謂)"
        )
    ]

    for pattern in explicit_patterns:
        for match in pattern.finditer(content):
            law_name, raw_article, item_block, subitem_block = match.groups()[:4]
            full_law = alias_map.get(law_name, law_name)
            article = normalize_article(raw_article)

            paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
            subparagraphs = re.findall(r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)", subitem_block or "")
            subparagraphs = [s.replace("或", "") for s in subparagraphs] if subparagraphs else [None]

            ref_type = (
                "內部引用" if full_law == current_law else
                "主法援引" if current_law in LAW_RELATIONS and full_law == LAW_RELATIONS[current_law]
                else "跨法源"
            )

            for p in paragraphs:
                for s in subparagraphs:
                    results.append({
                        "law_name": full_law,
                        "article": article,
                        "paragraph": p,
                        "subparagraph": s,
                        "type": ref_type
                    })

    # fallback：依第○○條（未提法名）
    fallback_pattern = re.compile(
        r"(?:依|根據|按|參照)[\s「『]*"
        r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
        r"((?:第[一二三四五六七八九十百\d]+項)*)?"
        r"((?:第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)*)?"
    )

    for match in fallback_pattern.finditer(content):
        raw_article, item_block, subitem_block = match.groups()
        article = normalize_article(raw_article)

        paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
        subparagraphs = re.findall(r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)", subitem_block or "")
        subparagraphs = [s.replace("或", "") for s in subparagraphs] if subparagraphs else [None]

        for p in paragraphs:
            for s in subparagraphs:
                results.append({
                    "law_name": current_law,
                    "article": article,
                    "paragraph": p,
                    "subparagraph": s,
                    "type": "內部引用"
                })

    # 隱性條號（無引言詞）
    implicit_pattern = re.compile(
        r"(?<!法)(?<![\u4e00-\u9fa5])(?<!依)(?<!根據)(?<!按)(?<!參照)"
        r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
        r"((?:第[一二三四五六七八九十百\d]+項)*)?"
        r"((?:第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)*)?"
    )

    for match in implicit_pattern.finditer(content):
        raw_article, item_block, subitem_block = match.groups()
        article = normalize_article(raw_article)

        paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
        subparagraphs = re.findall(r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)", subitem_block or "")
        subparagraphs = [s.replace("或", "") for s in subparagraphs] if subparagraphs else [None]

        for p in paragraphs:
            for s in subparagraphs:
                results.append({
                    "law_name": current_law,
                    "article": article,
                    "paragraph": p,
                    "subparagraph": s,
                    "type": "內部引用"
                })

    return results


# def extract_references(content, current_law):
#     if not content:
#         return []

#     results = []

#     alias_map = {
#         "本法": LAW_RELATIONS.get(current_law, current_law),
#         "本細則": current_law
#     }

#     explicit_patterns = [
#         re.compile(
#             r"(?:依|根據|按|參照)[\s「『]*"
#             r"(本法|本細則|[\u4e00-\u9fa5]{2,6}法)"
#             r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
#             r"((?:第[一二三四五六七八九十百\d]+項)*)?"
#             r"((?:第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)*)?"
#         ),
#         re.compile(
#             r"(本法|本細則|[\u4e00-\u9fa5]{2,}法)"
#             r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
#             r"(第[一二三四五六七八九十百\d]+項)?"
#             r"(第[一二三四五六七八九十百\d]+款)?"
#             r"(所稱|所定|所謂)"
#         )
#     ]

#     for pattern in explicit_patterns:
#         for match in pattern.finditer(content):
#             law_name, article, item_block, subitem_block = match.groups()[:4]
#             full_law = alias_map.get(law_name, law_name)

#             paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
#             subparagraphs = re.findall(r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)", subitem_block or "")
#             subparagraphs = [s.replace("或", "") for s in subparagraphs] if subparagraphs else [None]

#             ref_type = (
#                 "內部引用" if full_law == current_law else
#                 "主法援引" if current_law in LAW_RELATIONS and full_law == LAW_RELATIONS[current_law]
#                 else "跨法源"
#             )

#             for p in paragraphs:
#                 for s in subparagraphs:
#                     results.append({
#                         "law_name": full_law,
#                         "article": article,
#                         "paragraph": p,
#                         "subparagraph": s,
#                         "type": ref_type
#                     })

#     # 隱性引用：預設為 current_law，標記為內部引用
#     implicit_pattern = re.compile(
#         r"(?<!法)(?<![\u4e00-\u9fa5])(?<!依)(?<!根據)(?<!按)(?<!參照)"
#         r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
#         r"((?:第[一二三四五六七八九十百\d]+項)*)?"
#         r"((?:第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)*)?"
#     )

#     for match in implicit_pattern.finditer(content):
#         article, item_block, subitem_block = match.groups()
#         paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
#         subparagraphs = re.findall(r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)", subitem_block or "")
#         subparagraphs = [s.replace("或", "") for s in subparagraphs] if subparagraphs else [None]

#         for p in paragraphs:
#             for s in subparagraphs:
#                 results.append({
#                     "law_name": current_law,
#                     "article": article,
#                     "paragraph": p,
#                     "subparagraph": s,
#                     "type": "內部引用"
#                 })

#     return results

def handle_law(json_data, index_name):
    is_chapter_based = "章節" in json_data
    documents = json_data.get("章節", []) if is_chapter_based else [{"章名": None, "條文": json_data.get("條文", [])}]

    if not documents or all(not chapter.get("條文") for chapter in documents):
        print("⚠️ 檔案缺少條文資料")
        return 0, 0

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body={
            "mappings": {
                "properties": {
                    "name": {"type": "keyword"},
                    "date": {"type": "keyword"},
                    "chapter_title": {"type": "keyword"},
                    "number": {"type": "keyword"},
                    "law_type": {"type": "keyword"},
                    "law_family": {"type": "keyword"},
                    "is_supplement": {"type": "boolean"},
                    "parent_law": {
                        "properties": {
                            "name": {"type": "keyword"},
                            "article_links": {
                                "type": "nested",
                                "properties": {
                                    "article": {"type": "keyword"},
                                    "paragraph": {"type": "keyword"},
                                    "subparagraph": {"type": "keyword"}
                                }
                            }
                        }
                    },
                    "reference_laws": {
                        "type": "nested",
                        "properties": {
                            "law_name": {"type": "keyword"},
                            "article": {"type": "keyword"},
                            "paragraph": {"type": "keyword"},
                            "subparagraph": {"type": "keyword"},
                            "type": {"type": "keyword"}
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

            law_name = json_data.get("法規名稱", "")
            reference_laws = extract_references(content, law_name)
            law_type = "細則" if law_name in LAW_RELATIONS else "主法"
            law_family = LAW_FAMILY.get(law_name, "其他")

            # parent_law only for 細則
            parent_law = None
            if law_type == "細則":
                article_links = [
                    {
                        "article": ref["article"],
                        "paragraph": ref.get("paragraph"),
                        "subparagraph": ref.get("subparagraph")
                    }
                    for ref in reference_laws
                    if ref["law_name"] == LAW_RELATIONS[law_name]
                ]
                parent_law = {
                    "name": LAW_RELATIONS[law_name],
                    "article_links": article_links
                } if article_links else None

            if es.exists(index=index_name, id=doc_id):
                continue

            try:
                es.index(index=index_name, id=doc_id, body={
                    "name": law_name,
                    "date": json_data.get("修正日期", ""),
                    "chapter_title": chapter_title,
                    "number": clause_id,
                    "law_type": law_type,
                    "law_family": law_family,
                    "is_supplement": law_type == "細則",
                    "parent_law": parent_law,
                    "reference_laws": reference_laws,
                    "content": content,
                    "embedding": embedding
                })
                success += 1
            except Exception as e:
                print(f"❌ 寫入失敗：{clause_id} - {e}")
                failed += 1

    return success, failed

# 資料目錄處理邏輯
DATA_DIR = "data"
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

for filename in files:
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"🔴 無法讀取 {filename}：{e}")
        continue

    base = os.path.splitext(filename)[0]
    index_name = f"ai_{base}_index"
    print(f"\n📄 處理檔案：{filename} → Index: {index_name}")

    if any(k in base for k in ["immigration-law", "immigration-regulations", "nationality-law"]):
        success, failed = handle_law(json_data, index_name)
    else:
        print(f"⚠️ 不支援的檔案格式：{filename}")
        continue

    print(f"✅ 寫入成功：{success} 筆，⚠️ 失敗：{failed} 筆")

print("\n📌 所有檔案處理完成！")








# from elasticsearch import Elasticsearch, ConnectionError
# from sentence_transformers import SentenceTransformer  # 向量化模型
# from dotenv import load_dotenv
# import os
# import json
# import uuid
# import re

# load_dotenv()

# ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")

# """ 連接到 ES """
# try:
#     es = Elasticsearch(ES_HOST)

#     # 測試連線
#     if not es.ping():
#         raise ConnectionError("❌ 無法連接到 Elasticsearch，請檢查伺服器是否正在運行。")

#     print("✅ 成功連接到 Elasticsearch")
# except ConnectionError as e:
#     print(f"🔴 Elasticsearch 連線錯誤: {e}")
#     exit(1)  # 退出程式

# """ 初始化 embedding model """
# model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# # 主法－細則對應表
# LAW_RELATIONS = {
#     "入出國及移民法施行細則": "入出國及移民法",
#     "國籍法施行細則": "國籍法"
# }

# def extract_references(content, current_law):
#     if not content:
#         return []

#     results = []

#     if current_law in LAW_RELATIONS:
#         alias_map = {
#             "本法": LAW_RELATIONS[current_law],
#             "本細則": current_law
#         }
#     else:
#         alias_map = {
#             "本法": current_law,
#             "本細則": current_law
#         }

#     explicit_block_pattern = re.compile(
#         r"(?:依|根據|按|參照)[\s「『]*"
#         r"(本法|本細則|[\u4e00-\u9fa5]{2,6}法)"
#         r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
#         r"((?:第[一二三四五六七八九十百\d]+項(?:、第[一二三四五六七八九十百\d]+項)*))?"
#         r"((?:第[一二三四五六七八九十百\d]+款(?:、第[一二三四五六七八九十百\d]+款)*|或第[一二三四五六七八九十百\d]+款)*)?"
#     )

#     extra_reference_pattern = re.compile(
#         r"(本法|本細則|[\u4e00-\u9fa5]{2,}法)"
#         r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
#         r"(第[一二三四五六七八九十百\d]+項)?"
#         r"(第[一二三四五六七八九十百\d]+款)?"
#         r"(所稱|所定|所謂)"
#     )

#     for pattern in [explicit_block_pattern, extra_reference_pattern]:
#         for match in pattern.finditer(content):
#             law_name, article, item_block, subitem_block = match.groups()[:4]
#             full_law = alias_map.get(law_name, law_name)

#             paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
#             subparagraphs = re.findall(
#                 r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)",
#                 subitem_block or ""
#             )
#             subparagraphs = [s.replace("或", "") for s in subparagraphs] if subparagraphs else [None]

#             for p in paragraphs:
#                 for s in subparagraphs:
#                     ref_type = (
#                         "內部引用" if full_law == current_law
#                         else "主法援引" if current_law in LAW_RELATIONS and full_law == LAW_RELATIONS[current_law]
#                         else "跨法源"
#                     )
#                     results.append({
#                         "law_name": full_law,
#                         "article": article,
#                         "paragraph": p,
#                         "subparagraph": s,
#                         "type": ref_type
#                     })

#     implicit_pattern = re.compile(
#         r"(?<!法)(?<![\u4e00-\u9fa5])(?<!依)(?<!根據)(?<!按)(?<!參照)"
#         r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
#         r"((?:第[一二三四五六七八九十百\d]+項(?:、第[一二三四五六七八九十百\d]+項)*))?"
#         r"((?:第[一二三四五六七八九十百\d]+款(?:、第[一二三四五六七八九十百\d]+款)*|或第[一二三四五六七八九十百\d]+款)*)?"
#     )

#     for match in implicit_pattern.finditer(content):
#         article, item_block, subitem_block = match.groups()

#         paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
#         subparagraphs = re.findall(
#             r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)",
#             subitem_block or ""
#         )
#         subparagraphs = [s.replace("或", "") for s in subparagraphs] if subparagraphs else [None]

#         for p in paragraphs:
#             for s in subparagraphs:
#                 results.append({
#                     "law_name": current_law,
#                     "article": article,
#                     "paragraph": p,
#                     "subparagraph": s,
#                     "type": "內部引用"
#                 })

#     return results


# # def extract_references(content, current_law):
# #     if not content:
# #         return []

# #     results = []

# #     if current_law in LAW_RELATIONS:
# #         alias_map = {
# #             "本法": LAW_RELATIONS[current_law],
# #             "本細則": current_law
# #         }
# #     else:
# #         alias_map = {
# #             "本法": current_law,
# #             "本細則": current_law
# #         }

# #     explicit_block_pattern = re.compile(
# #         r"(?:依|根據|按|參照)[\s「『]*"
# #         r"(本法|本細則|[\u4e00-\u9fa5]{2,6}法)"
# #         r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
# #         r"((?:第[一二三四五六七八九十百\d]+項(?:、第[一二三四五六七八九十百\d]+項)*))?"
# #         r"((?:第[一二三四五六七八九十百\d]+款(?:、第[一二三四五六七八九十百\d]+款)*|或第[一二三四五六七八九十百\d]+款)*)?"
# #     )

# #     extra_reference_pattern = re.compile(
# #         r"(本法|本細則|[\u4e00-\u9fa5]{2,}法)"
# #         r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
# #         r"(第[一二三四五六七八九十百\d]+項)?"
# #         r"(第[一二三四五六七八九十百\d]+款)?"
# #         r"(所稱|所定|所謂)"
# #     )

# #     for pattern in [explicit_block_pattern, extra_reference_pattern]:
# #         for match in pattern.finditer(content):
# #             law_name, article, item_block, subitem_block = match.groups()[:4]
# #             full_law = alias_map.get(law_name, law_name)

# #             paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
# #             subparagraphs = re.findall(
# #                 r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)",
# #                 subitem_block or ""
# #             )
# #             subparagraphs = [s.replace("或", "") for s in subparagraphs] if subparagraphs else [None]

# #             for p in paragraphs:
# #                 for s in subparagraphs:
# #                     results.append({
# #                         "law_name": full_law,
# #                         "article": article,
# #                         "paragraph": p,
# #                         "subparagraph": s
# #                     })

# #     implicit_pattern = re.compile(
# #         r"(?<!法)(?<![\u4e00-\u9fa5])(?<!依)(?<!根據)(?<!按)(?<!參照)"
# #         r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
# #         r"((?:第[一二三四五六七八九十百\d]+項(?:、第[一二三四五六七八九十百\d]+項)*))?"
# #         r"((?:第[一二三四五六七八九十百\d]+款(?:、第[一二三四五六七八九十百\d]+款)*|或第[一二三四五六七八九十百\d]+款)*)?"
# #     )

# #     for match in implicit_pattern.finditer(content):
# #         article, item_block, subitem_block = match.groups()

# #         paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
# #         subparagraphs = re.findall(
# #             r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)",
# #             subitem_block or ""
# #         )
# #         subparagraphs = [s.replace("或", "") for s in subparagraphs] if subparagraphs else [None]

# #         for p in paragraphs:
# #             for s in subparagraphs:
# #                 results.append({
# #                     "law_name": current_law,
# #                     "article": article,
# #                     "paragraph": p,
# #                     "subparagraph": s
# #                 })

# #     return results

# """ 處理 immigration-law.json 格式 """
# def handle_law(json_data, index_name):
#     # 判斷是否為章節制
#     is_chapter_based = "章節" in json_data

#     # 根據結構取得條文清單
#     if is_chapter_based:
#         documents = json_data.get("章節", [])
#     else:
#         documents = [{"章名": None, "條文": json_data.get("條文", [])}]

#     if not documents or all(not chapter.get("條文") for chapter in documents):
#         print(f"⚠️ 檔案缺少條文資料")
#         return 0, 0

#     # 建立 index（若尚未存在）
#     if not es.indices.exists(index=index_name):
#         es.indices.create(index=index_name, body={
#             "mappings": {
#                 "properties": {
#                     "name": {"type": "keyword"},
#                     "date": {"type": "keyword"},
#                     "chapter_title": {"type": "keyword"},
#                     "number": {"type": "keyword"},
#                     "reference_laws": {
#                         "type": "nested",
#                         "properties": {
#                             "law_name": {"type": "keyword"},
#                             "article": {"type": "keyword"},
#                             "paragraph": {"type": "keyword"},
#                             "subparagraph": {"type": "keyword"}
#                         }
#                     },
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
#         chapter_title = chapter.get("章名", "") or ""
#         for clause in chapter.get("條文", []):
#             clause_id = clause.get("條號")
#             content = clause.get("內容", "")

#             if not clause_id or not content:
#                 failed += 1
#                 continue

#             embedding = model.encode(content).tolist()
#             doc_id = f"{chapter_title}_{clause_id}".replace(" ", "")

#             # 參照法源
#             reference_laws = extract_references(content, json_data.get("法規名稱", ""))

#             # 若已存在就跳過處理
#             if es.exists(index=index_name, id=doc_id):
#                 continue

#             try:
#                 es.index(index=index_name, id=doc_id, body={
#                     "name": json_data.get("法規名稱", ""),
#                     "date": json_data.get("修正日期", ""),
#                     "chapter_title": chapter_title,
#                     "number": clause_id,
#                     "reference_laws": reference_laws, 
#                     "content": content,
#                     "embedding": embedding
#                 })
#                 success += 1
#             except Exception as e:
#                 print(f"❌ 寫入失敗：{clause_id} - {e}")
#                 failed += 1

#     return success, failed


# """ 在這裡增加其他處理格式 """


# """ 資料來源路徑 """
# DATA_DIR = "data"
# files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

# """ 根據檔名選擇處理邏輯 """
# for filename in files:
#     path = os.path.join(DATA_DIR, filename)
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             json_data = json.load(f)
#     except Exception as e:
#         print(f"🔴 無法讀取 {filename}：{e}")
#         continue

#     # 自動用檔名轉 index 名
#     base = os.path.splitext(filename)[0]  # e.g. "labor-law"
#     index_name = f"ai_{base}_index"

#     print(f"\n📄 處理檔案：{filename} → Index: {index_name}")

#     # 根據檔案名稱決定格式處理方式
#     # if "labor-law" in base:
#     #     success, failed = handle_labor_law(json_data, index_name)
#     if any(keyword in base for keyword in ["immigration-law", "immigration-regulations", "nationality-law"]):
#         success, failed = handle_law(json_data, index_name)
#     else:
#         print(f"⚠️ 尚未支援此檔案格式：{filename}")
#         continue

#     print(f"✅ 寫入成功：{success} 筆，⚠️ 失敗：{failed} 筆")

# print("\n📌 所有檔案處理完成！")




