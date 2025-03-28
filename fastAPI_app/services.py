from rag_fastapi import rag_fastapi # from workdir 底下的路徑檔案 import

def process_rag_query(query: str) -> str:
    """ 處理 RAG 查詢請求 """
    return rag_fastapi(query)