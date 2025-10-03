# from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from typing import Dict, Any, List
# import json
# import hashlib
# from datetime import datetime

# from app.config import settings, DBType
# from app.auth import get_current_active_user, require_admin, User
# from app.audio.transcriber import AudioTranscriber
# from app.llm.groq_client import GroqClient
# from app.vector_db.pinecone_client import PineconeClient
# from app.database.introspection import SchemaIntrospector
# from app.query.spec_validator import QueryValidator, QuerySpec
# from app.query.translator import QueryTranslator, MongoQueryTranslator
# from app.database.connection import get_database_connection

# app = FastAPI(title="SaaS Chatbot API", version="1.0.0")

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global components
# transcriber = AudioTranscriber()
# groq_client = GroqClient()
# pinecone_client = PineconeClient()
# query_validator = QueryValidator()

# # In-memory session storage (replace with Redis in production)
# sessions = {}
# audit_logs = []

# @app.post("/transcribe")
# async def transcribe_audio(
#     file: UploadFile = File(...),
#     session_id: str = Form(...),
#     role: str = Form("user"),
#     current_user: User = Depends(get_current_active_user)
# ):
#     """Transcribe audio file to English text"""
#     if not file.content_type.startswith("audio/"):
#         raise HTTPException(400, "Invalid audio file")
    
#     # Check file size
#     file.file.seek(0, 2)  # Seek to end
#     file_size = file.file.tell()
#     file.file.seek(0)  # Reset to beginning
    
#     if file_size > settings.MAX_AUDIO_SIZE:
#         raise HTTPException(400, "Audio file too large")
    
#     try:
#         text, confidence = await transcriber.transcribe_audio(file)
        
#         # Store transcript in session memory
#         if session_id not in sessions:
#             sessions[session_id] = []
        
#         sessions[session_id].append({
#             "role": role,
#             "type": "audio_transcript",
#             "content": text,
#             "timestamp": datetime.utcnow(),
#             "confidence": confidence
#         })
        
#         return {
#             "text": text,
#             "confidence": confidence,
#             "session_id": session_id
#         }
    
#     except Exception as e:
#         raise HTTPException(500, f"Transcription failed: {str(e)}")

# @app.post("/query")
# async def process_query(
#     session_id: str = Form(...),
#     role: str = Form("user"),
#     text: str = Form(...),
#     current_user: User = Depends(get_current_active_user)
# ):
#     """Process natural language query and return results"""
#     try:
#         # 1. Retrieve session memory
#         conversation_history = sessions.get(session_id, [])
        
#         # 2. Get relevant schema docs from Pinecone
#         schema_context = await _get_relevant_schemas(text, settings.DB_TYPE.value)
        
#         # 3. Generate query spec using Groq LLM
#         history_text = _format_conversation_history(conversation_history)
#         query_spec_dict = groq_client.generate_query_spec(text, schema_context, history_text)
        
#         # 4. Validate query spec
#         query_spec = query_validator.validate_spec(query_spec_dict)
#         if not query_validator.is_safe_query(query_spec):
#             raise HTTPException(400, "Query contains unsafe operations")
        
#         # 5. Translate to database query
#         if settings.DB_TYPE == DBType.MONGODB:
#             translator = MongoQueryTranslator()
#             query = translator.spec_to_mongo(query_spec)
#         else:
#             translator = QueryTranslator(settings.DB_TYPE)
#             query = translator.spec_to_sql(query_spec)
        
#         # 6. Execute query with read-only user
#         db_conn = get_database_connection(settings.DB_TYPE, read_only=True)
#         results = await db_conn.execute_query(query)
        
#         # 7. Generate summary
#         summary = groq_client.generate_summary(text, results)
        
#         # 8. Audit logging
#         audit_entry = {
#             "timestamp": datetime.utcnow(),
#             "user": current_user.username,
#             "session_id": session_id,
#             "query_text": text,
#             "query_spec": query_spec_dict,
#             "executed_query": str(query),
#             "results_count": len(results),
#             "query_hash": hashlib.md5(str(query).encode()).hexdigest()
#         }
#         audit_logs.append(audit_entry)
        
#         # 9. Update session memory
#         if session_id not in sessions:
#             sessions[session_id] = []
            
#         sessions[session_id].append({
#             "role": role,
#             "type": "query",
#             "content": text,
#             "timestamp": datetime.utcnow()
#         })
#         sessions[session_id].append({
#             "role": "assistant",
#             "type": "response",
#             "content": summary,
#             "results": results[:10],  # Store first 10 rows
#             "timestamp": datetime.utcnow()
#         })
        
#         return {
#             "result_table": results,
#             "summary": summary,
#             "session_id": session_id
#         }
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(500, f"Query processing failed: {str(e)}")

# @app.post("/reindex-schema")
# async def reindex_schema(current_user: User = Depends(require_admin)):
#     """Reindex database schema in Pinecone (admin only)"""
#     try:
#         schema_introspector = SchemaIntrospector(settings.DB_TYPE)
        
#         # Delete old embeddings
#         pinecone_client.delete_schema_embeddings(settings.DB_TYPE.value)
        
#         # Introspect and store new embeddings
#         schema_docs = schema_introspector.introspect_schema()
#         pinecone_client.store_schema_embeddings(schema_docs, settings.DB_TYPE.value)
        
#         return {
#             "message": "Schema reindexed successfully",
#             "tables_processed": len(schema_docs)
#         }
    
#     except Exception as e:
#         raise HTTPException(500, f"Schema reindexing failed: {str(e)}")

# @app.get("/session/{session_id}")
# async def get_session_history(
#     session_id: str,
#     current_user: User = Depends(get_current_active_user)
# ):
#     """Get conversation history for session"""
#     return sessions.get(session_id, [])

# @app.get("/audit-logs")
# async def get_audit_logs(current_user: User = Depends(require_admin)):
#     """Get audit logs (admin only)"""
#     return audit_logs

# async def _get_relevant_schemas(query: str, db_type: str) -> str:
#     """Get relevant schema documents from Pinecone"""
#     results = pinecone_client.search_similar_schemas(query, db_type, k=3)
#     return "\n\n".join([doc.page_content for doc in results])

# def _format_conversation_history(history: List[Dict]) -> str:
#     """Format conversation history for LLM context"""
#     return "\n".join([
#         f"{item['role']}: {item['content']}" 
#         for item in history[-6:]  # Last 6 messages
#     ])

# # WebSocket for streaming responses (optional)
# @app.websocket("/ws/{session_id}")
# async def websocket_endpoint(websocket: WebSocket, session_id: str):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_text()
#             # Implement streaming logic here
#             await websocket.send_text(f"Message received: {data}")
#     except Exception as e:
#         print(f"WebSocket error: {e}")
#     finally:
#         await websocket.close()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import json
import hashlib
from datetime import datetime

from app.config import settings, DBType
from app.auth import get_current_active_user, require_admin, User, router as auth_router
from app.audio.transcriber import AudioTranscriber
from app.llm.groq_client import GroqClient
from app.vector_db.pinecone_client import PineconeClient
from app.database.introspection import SchemaIntrospector
from app.query.spec_validator import QueryValidator, QuerySpec
from app.query.translator import QueryTranslator, MongoQueryTranslator
from app.database.connection import get_database_connection

app = FastAPI(title="SaaS Chatbot API", version="1.0.0")

# Include auth routes
app.include_router(auth_router, prefix="/auth", tags=["authentication"])

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
transcriber = AudioTranscriber()
groq_client = GroqClient()
pinecone_client = PineconeClient()
query_validator = QueryValidator()

# In-memory session storage (replace with Redis in production)
sessions = {}
audit_logs = []

@app.get("/")
async def root():
    return {"message": "SaaS Chatbot API", "version": "1.0.0"}

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    role: str = Form("user"),
    current_user: User = Depends(get_current_active_user)
):
    """Transcribe audio file to English text"""
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Invalid audio file")
    
    try:
        text, confidence = await transcriber.transcribe_audio(file)
        
        # Store transcript in session memory
        if session_id not in sessions:
            sessions[session_id] = []
        
        sessions[session_id].append({
            "role": role,
            "type": "audio_transcript",
            "content": text,
            "timestamp": datetime.utcnow(),
            "confidence": confidence
        })
        
        return {
            "text": text,
            "confidence": confidence,
            "session_id": session_id
        }
    
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")

@app.post("/query")
async def process_query(
    session_id: str = Form(...),
    role: str = Form("user"),
    text: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """Process natural language query and return results"""
    try:
        # 1. Retrieve session memory
        conversation_history = sessions.get(session_id, [])
        
        # 2. Get relevant schema docs from Pinecone
        schema_context = await _get_relevant_schemas(text, settings.DB_TYPE.value)
        
        # 3. Generate query spec using Groq LLM
        history_text = _format_conversation_history(conversation_history)
        query_spec_dict = groq_client.generate_query_spec(text, schema_context, history_text)
        
        # 4. Validate query spec
        query_spec = query_validator.validate_spec(query_spec_dict)
        if not query_validator.is_safe_query(query_spec):
            raise HTTPException(400, "Query contains unsafe operations")
        
        # 5. Translate to database query
        if settings.DB_TYPE == DBType.MONGODB:
            translator = MongoQueryTranslator()
            query = translator.spec_to_mongo(query_spec)
        else:
            translator = QueryTranslator(settings.DB_TYPE)
            query = translator.spec_to_sql(query_spec)
        
        # 6. Execute query with read-only user
        db_conn = get_database_connection(settings.DB_TYPE, read_only=True)
        results = await db_conn.execute_query(query)
        
        # 7. Generate summary
        summary = groq_client.generate_summary(text, results)
        
        # 8. Audit logging
        audit_entry = {
            "timestamp": datetime.utcnow(),
            "user": current_user.username,
            "session_id": session_id,
            "query_text": text,
            "query_spec": query_spec_dict,
            "executed_query": str(query),
            "results_count": len(results),
            "query_hash": hashlib.md5(str(query).encode()).hexdigest()
        }
        audit_logs.append(audit_entry)
        
        # 9. Update session memory
        if session_id not in sessions:
            sessions[session_id] = []
            
        sessions[session_id].append({
            "role": role,
            "type": "query",
            "content": text,
            "timestamp": datetime.utcnow()
        })
        sessions[session_id].append({
            "role": "assistant",
            "type": "response",
            "content": summary,
            "results": results[:10],
            "timestamp": datetime.utcnow()
        })
        
        return {
            "result_table": results,
            "summary": summary,
            "session_id": session_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Query processing failed: {str(e)}")

@app.post("/reindex-schema")
async def reindex_schema(current_user: User = Depends(require_admin)):
    """Reindex database schema in Pinecone (admin only)"""
    try:
        schema_introspector = SchemaIntrospector(settings.DB_TYPE)
        
        # Delete old embeddings
        pinecone_client.delete_schema_embeddings(settings.DB_TYPE.value)
        
        # Introspect and store new embeddings
        schema_docs = schema_introspector.introspect_schema()
        pinecone_client.store_schema_embeddings(schema_docs, settings.DB_TYPE.value)
        
        return {
            "message": "Schema reindexed successfully",
            "tables_processed": len(schema_docs)
        }
    
    except Exception as e:
        raise HTTPException(500, f"Schema reindexing failed: {str(e)}")

@app.get("/session/{session_id}")
async def get_session_history(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get conversation history for session"""
    return sessions.get(session_id, [])

@app.get("/audit-logs")
async def get_audit_logs(current_user: User = Depends(require_admin)):
    """Get audit logs (admin only)"""
    return audit_logs

async def _get_relevant_schemas(query: str, db_type: str) -> str:
    """Get relevant schema documents from Pinecone"""
    try:
        results = pinecone_client.search_similar_schemas(query, db_type, k=3)
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        print(f"Error retrieving schemas: {e}")
        return "No schema context available"

def _format_conversation_history(history: List[Dict]) -> str:
    """Format conversation history for LLM context"""
    return "\n".join([
        f"{item['role']}: {item['content']}" 
        for item in history[-6:]
    ])

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)