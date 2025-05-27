from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from services import process_rag_query  # from workdir 底下的路徑檔案 import
from dotenv import load_dotenv
from pathlib import Path
import os

app = FastAPI()

load_dotenv()


API_TOKEN = os.getenv("API_TOKEN")
print(f"🔐 Expected apitoken: '{API_TOKEN}'")


class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
# async def rag_query(request: Request, query: QueryRequest, authorization: str = Header(None)):
#     if not authorization or not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
#     token = authorization.replace("Bearer ", "")
#     print(f"🔐 Expected token: '{token}'")
#     if token != API_TOKEN:
#         raise HTTPException(status_code=401, detail="Invalid token")

#     response = process_rag_query(query.query)
#     return {"response": response}

async def rag_query(request: QueryRequest):
    response = process_rag_query(request.query)  
    return {"response": response}