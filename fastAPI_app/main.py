from fastapi import FastAPI
from pydantic import BaseModel
from services import process_rag_query  # from workdir 底下的路徑檔案 import

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
async def rag_query(request: QueryRequest):
    response = process_rag_query(request.query)  
    return {"response": response}