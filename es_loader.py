from elasticsearch import Elasticsearch, ConnectionError, helpers
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime

import os
import json
import re

load_dotenv()

ES_HOST = os.getenv("ELASTICSEARCH_HOST")
ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

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

# 連線檢查
try:
    es = Elasticsearch(ES_HOST, api_key=ES_API_KEY, verify_certs=True)
    if not es.ping():
        raise ConnectionError("❌ 無法連接到 Elasticsearch。")
    print("✅ 成功連接到 Elasticsearch")
except ConnectionError as e:
    print(f"🔴 連線錯誤: {e}")
    exit(1)

# 向量模型
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# 中文數字轉整數
CHINESE_NUMERAL_MAP = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000
}

def chinese_numeral_to_int(text):
    result = 0
    tmp = 0
    units = {"十": 10, "百": 100, "千": 1000}
    for ch in text:
        if ch in units:
            unit = units[ch]
            if tmp == 0:
                tmp = 1
            result += tmp * unit
            tmp = 0
        else:
            tmp = CHINESE_NUMERAL_MAP.get(ch, 0)
    result += tmp
    return result

def normalize_article(article_text):
    m = re.match(r"第([一二三四五六七八九十百零\d]+)條", article_text)
    if m:
        arabic = chinese_numeral_to_int(m.group(1))
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

    # explicit patterns
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
            subparagraphs = re.findall(
                r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)",
                subitem_block or ""
            ) or [None]
            subparagraphs = [s.replace("或", "") if s else None for s in subparagraphs]

            ref_type = (
                "內部引用" if full_law == current_law else
                "主法援引" if LAW_RELATIONS.get(current_law) == full_law else
                "跨法源"
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

    # fallback
    fallback = re.compile(
        r"(?:依|根據|按|參照)[\s「『]*"
        r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
        r"((?:第[一二三四五六七八九十百\d]+項)*)?"
        r"((?:第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)*)?"
    )
    for match in fallback.finditer(content):
        raw, item_block, subitem_block = match.groups()
        article = normalize_article(raw)
        paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
        subparagraphs = re.findall(
            r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)",
            subitem_block or ""
        ) or [None]
        subparagraphs = [s.replace("或", "") if s else None for s in subparagraphs]
        for p in paragraphs:
            for s in subparagraphs:
                results.append({
                    "law_name": current_law,
                    "article": article,
                    "paragraph": p,
                    "subparagraph": s,
                    "type": "內部引用"
                })

    # implicit
    implicit = re.compile(
        r"(?<!法)(?<![\u4e00-\u9fa5])(?<!依)(?<!根據)(?<!按)(?<!參照)"
        r"(第[一二三四五六七八九十百\d]+條(?:之[\d]+)?)"
        r"((?:第[一二三四五六七八九十百\d]+項)*)?"
        r"((?:第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)*)?"
    )
    for match in implicit.finditer(content):
        raw, item_block, subitem_block = match.groups()
        article = normalize_article(raw)
        paragraphs = re.findall(r"第[一二三四五六七八九十百\d]+項", item_block or "") or [None]
        subparagraphs = re.findall(
            r"(第[一二三四五六七八九十百\d]+款|或第[一二三四五六七八九十百\d]+款)",
            subitem_block or ""
        ) or [None]
        subparagraphs = [s.replace("或", "") if s else None for s in subparagraphs]
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

def handle_law(json_data, index_name):
    is_chapter_based = "章節" in json_data
    documents = json_data.get("章節", []) if is_chapter_based else [{"章名": None, "條文": json_data.get("條文", [])}]
    if not documents or all(not c.get("條文") for c in documents):
        print("⚠️ 檔案缺少條文資料")
        return 0, 0

    law_name = json_data.get("法規名稱", "")
    law_type = "細則" if law_name in LAW_RELATIONS else "主法"
    law_family = LAW_FAMILY.get(law_name, "其他")

    success = failed = 0

    for chapter in documents:
        chap_title = chapter.get("章名") or ""
        for clause in chapter.get("條文", []):
            num = clause.get("條號")
            content = clause.get("內容", "")
            if not num or not content:
                failed += 1
                continue

            try:
                emb = model.encode(content).tolist()
            except Exception as e:
                print(f"❌ 向量轉換失敗：{num} - {e}")
                failed += 1
                continue

            doc_id = f"{chap_title}_{num}".replace(" ", "")
            refs = extract_references(content, law_name)

            parent = None
            if law_type == "細則":
                links = [
                    {"article": r["article"], "paragraph": r["paragraph"], "subparagraph": r["subparagraph"]}
                    for r in refs if r["law_name"] == LAW_RELATIONS[law_name]
                ]
                if links:
                    parent = {"name": LAW_RELATIONS[law_name], "article_links": links}

            try:
                es.index(
                    index=index_name,
                    id=doc_id,
                    document={
                        "name": law_name,
                        "date": json_data.get("修正日期", ""),
                        "chapter_title": chap_title,
                        "number": num,
                        "law_type": law_type,
                        "law_family": law_family,
                        "is_supplement": law_type == "細則",
                        "parent_law": parent,
                        "reference_laws": refs,
                        "content": content,
                        "embedding": emb,
                        "@timestamp": datetime.utcnow().isoformat()
                    }
                )
                success += 1
            except Exception as e:
                print(f"❌ 寫入失敗：{num} - {type(e).__name__} - {e}")
                if hasattr(e, 'meta'):
                    print("🔍 meta status:", getattr(e.meta, 'status', 'N/A'))
                    print("🔍 meta url:", getattr(e.meta, 'url', 'N/A'))
                    print("🔍 meta headers:", getattr(e.meta, 'headers', 'N/A'))
                if hasattr(e, 'body'):
                    print("🔍 body:", e.body)
                failed += 1

    return success, failed

if __name__ == "__main__":
    DATA_DIR = "data"
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]

    for fn in files:
        path = os.path.join(DATA_DIR, fn)
        try:
            with open(path, encoding="utf-8") as f:
                jd = json.load(f)
        except Exception as e:
            print(f"🔴 無法讀取 {fn}：{e}")
            continue

        base = os.path.splitext(fn)[0]               # e.g. "immigration-law"
        index_name = f"ai-laws-{base}"               # 對應你事先在 DevTools 建的 ai-laws-* index
        print(f"\n📄 處理檔案：{fn} → Index: {index_name}")

        if any(k in base for k in ["immigration-law", "immigration-regulations", "nationality-law"]):
            s, f = handle_law(jd, index_name)
            print(f"✅ 寫入成功：{s} 筆，⚠️ 失敗：{f} 筆")
        else:
            print(f"⚠️ 不支援的檔案格式：{fn}")

    print("\n📌 所有檔案處理完成！")