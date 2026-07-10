import os
import traceback
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from graph_workflow import app_graph, fetch_schema
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# =========================================================
# 1️⃣ ENVIRONMENT & APP SETUP
# =========================================================
load_dotenv()

app = FastAPI(title="Dynamic LangGraph Database AI Backend", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2️⃣ SCHEMAS
# =========================================================
class DBConfig(BaseModel):
    db_host: str
    db_user: str
    db_password: str
    db_name: str

class QueryRequest(BaseModel):
    query: str
    config: DBConfig

class IndexRequest(BaseModel):
    config: DBConfig

# =========================================================
# 3️⃣ API: SCHEMA INDEXING (for RAG)
# =========================================================
@app.post("/index-schema")
async def index_schema(req: IndexRequest):
    try:
        config_dict = req.config.dict()
        
        # Verify DB connection and fetch schema
        schema = fetch_schema(config_dict)
        docs = [json.dumps(item) for item in schema]
        
        # Index in Pinecone
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            return {"message": "✅ Schema fetched, but Pinecone API key not found in backend .env for vector indexing.", "tables": len(set([s["TABLE_NAME"] or s["table_name"] for s in schema]))}
            
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        pc = Pinecone(api_key=pinecone_key)
        index_name = os.getenv("PINECONE_INDEX_NAME", "dynamic_schema_index")
        
        if index_name not in [i["name"] for i in pc.list_indexes()]:
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        pinecone_index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
        vector_store.add_texts(docs)
        
        return {"message": "✅ Schema indexed into Pinecone", "tables": len(set([s["TABLE_NAME"] or s["table_name"] for s in schema]))}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# 4️⃣ API: LANGGRAPH QUERY EXECUTION
# =========================================================
@app.post("/query")
async def query_database(req: QueryRequest):
    try:
        config_dict = req.config.dict()
        user_query = req.query.strip()
        
        # Initialize state
        initial_state = {
            "query": user_query,
            "config": config_dict,
            "schema_context": "",
            "pinecone_context": "",
            "sql_query": "",
            "columns": [],
            "rows": [],
            "is_chart": False,
            "chart_type": "",
            "error": "",
            "retries": 0
        }
        
        # Invoke LangGraph
        result = app_graph.invoke(initial_state)
        
        if result.get("error"):
            raise HTTPException(status_code=400, detail=f"Query failed after retries. Error: {result['error']}")
            
        return {
            "query": user_query,
            "sql": result.get("sql_query"),
            "columns": result.get("columns", []),
            "rows": result.get("rows", []),
            "is_chart": result.get("is_chart", False),
            "chart_type": result.get("chart_type")
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Query failed: {str(e)}")

# =========================================================
# 5️⃣ HEALTH CHECK
# =========================================================
@app.get("/")
async def root():
    return {"status": "✅ Backend Running (LangGraph enabled)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
