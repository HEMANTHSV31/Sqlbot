# import os
# import time
# import logging
# from typing import List, Dict, Optional
# from dotenv import load_dotenv
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import mysql.connector
# from groq import Groq, GroqError
# import openai
# from pydantic import BaseModel

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Chatbot API", version="1.0.0")

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Database configuration
# DB_CONFIG = {
#     "host": os.getenv("DB_HOST", "localhost"),
#     "user": os.getenv("DB_USER", "root"),
#     "password": os.getenv("DB_PASSWORD", ""),
#     "database": os.getenv("DB_NAME", "chatbot_db"),
#     "port": int(os.getenv("DB_PORT", 3306)),  # Convert to int
# }

# # Initialize Groq client safely
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise RuntimeError("GROQ_API_KEY is not set in environment variables")
# groq_client = Groq(api_key=GROQ_API_KEY)

# # Initialize OpenAI client
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise RuntimeError("OPENAI_API_KEY is not set in environment variables")
# openai.api_key = OPENAI_API_KEY

# # Groq model
# GROQ_MODEL = "llama-3.1-8b-instant"

# # Pydantic models
# class TextQuery(BaseModel):
#     message: str

# class ChatResponse(BaseModel):
#     message: str
#     sql_query: Optional[str] = None
#     data: Optional[List[Dict]] = None
#     error: Optional[str] = None

# # System prompt for SQL generation
# SQL_SYSTEM_PROMPT = """
# You are a SQL query generator. Your task is to generate ONLY SELECT queries for the MySQL database.

# Available tables:
# - users(id, name, email, created_at)
# - products(id, name, sku, price, created_at)
# - orders(id, user_id, total, created_at)
# - order_items(id, order_id, product_id, quantity, price)
# - product_sales(id, product_id, sale_date, units_sold, revenue)

# Rules:
# 1. Generate ONLY SELECT queries.
# 2. Only use the tables listed above.
# 3. If the request requires data outside these tables, return this exact fallback query:
#    SELECT 'Invalid request. Only users, products, orders, order_items, and product_sales are supported.' AS error;
# 4. Prevent SQL injection - never use direct user input in queries.
# 5. If no table is explicitly mentioned, infer the best table based on context.
# 6. Always use proper JOINs when needed.
# 7. Include relevant WHERE clauses for filtering.
# 8. Use LIMIT for large result sets when appropriate.
# 9. Return ONLY the SQL query, nothing else.
# """

# # Database connection
# def get_db_connection():
#     try:
#         return mysql.connector.connect(**DB_CONFIG)
#     except mysql.connector.Error as e:
#         logger.error(f"Database connection error: {e}")
#         raise HTTPException(status_code=500, detail="Database connection failed")

# # Generate SQL query using Groq
# def generate_sql_query(user_message: str) -> str:
#     try:
#         completion = groq_client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": SQL_SYSTEM_PROMPT},
#                 {"role": "user", "content": user_message}
#             ],
#             model=GROQ_MODEL,
#             temperature=0.1,
#             max_tokens=500
#         )
#         sql_query = completion.choices[0].message.content.strip()
#         # Remove markdown code block
#         if sql_query.startswith("```sql"):
#             sql_query = sql_query[6:]
#         if sql_query.endswith("```"):
#             sql_query = sql_query[:-3]
#         return sql_query.strip()
#     except GroqError as e:
#         logger.error(f"Groq API error: {e}")
#         raise HTTPException(status_code=500, detail="Could not generate SQL query")

# # Execute SQL query
# def execute_sql_query(sql_query: str) -> List[Dict]:
#     conn = None
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor(dictionary=True)
#         cursor.execute(sql_query)
#         results = cursor.fetchall()
#         return results
#     except mysql.connector.Error as e:
#         logger.error(f"SQL execution error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if conn and conn.is_connected():
#             conn.close()

# # Transcribe audio
# async def transcribe_audio(audio_file: UploadFile) -> str:
#     try:
#         file_content = await audio_file.read()
#         transcript = openai.Audio.transcriptions.create(
#             model="whisper-1",
#             file=("audio.wav", file_content, "audio/wav")
#         )
#         return transcript.text
#     except Exception as e:
#         logger.error(f"Whisper API error: {e}")
#         raise HTTPException(status_code=500, detail="Could not transcribe audio")

# # Refine transcription
# def refine_transcription(transcript: str) -> str:
#     try:
#         completion = groq_client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": "Fix grammar and spelling errors while preserving meaning."},
#                 {"role": "user", "content": transcript}
#             ],
#             model=GROQ_MODEL,
#             temperature=0.1,
#             max_tokens=500
#         )
#         return completion.choices[0].message.content.strip()
#     except Exception as e:
#         logger.error(f"Transcription refinement error: {e}")
#         return transcript

# # Routes
# @app.post("/query-text", response_model=ChatResponse)
# async def query_text(query: TextQuery):
#     try:
#         sql_query = generate_sql_query(query.message)
#         results = execute_sql_query(sql_query)
#         message = results[0]["error"] if results and "error" in results[0] else f"Found {len(results)} result(s)."
#         return ChatResponse(message=message, sql_query=sql_query, data=results)
#     except HTTPException as he:
#         return ChatResponse(message=f"Error: {he.detail}", error=he.detail)

# @app.post("/query-audio", response_model=ChatResponse)
# async def query_audio(audio: UploadFile = File(...)):
#     if not audio.content_type.startswith("audio/"):
#         raise HTTPException(status_code=400, detail="File must be an audio file")
#     transcript = await transcribe_audio(audio)
#     if not transcript.strip():
#         raise HTTPException(status_code=400, detail="No speech detected in audio")
#     refined_transcript = refine_transcription(transcript)
#     sql_query = generate_sql_query(refined_transcript)
#     results = execute_sql_query(sql_query)
#     message = results[0]["error"] if results and "error" in results[0] else f"Found {len(results)} result(s)."
#     return ChatResponse(message=message, sql_query=sql_query, data=results)

# @app.get("/health")
# async def health_check():
#     try:
#         conn = get_db_connection()
#         conn.close()
#         return {"status": "healthy", "database": "connected"}
#     except Exception as e:
#         return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

# @app.get("/models")
# async def get_available_models():
#     try:
#         models = groq_client.models.list()
#         return {"available_models": [m.id for m in models.data]}
#     except Exception as e:
#         logger.error(f"Error fetching models: {e}")
#         return {"available_models": [], "error": str(e)}


import os
import logging
from typing import List, Dict, Optional
import io
import re

# --- Required for High-Performance Local Transcription ---
from faster_whisper import WhisperModel
import noisereduce as nr
import numpy as np
import torch
from pydub import AudioSegment
# --------------------------------------------------------

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from pydantic import BaseModel

# --- Final Corrected Imports ---
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. INITIAL SETUP ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Core SQL Chatbot API",
    version="4.0.0",
    description="A robust API focused on high-quality SQL generation using a local Whisper model, Pinecone, and Groq.",
)

# --- 2. CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. CONFIGURATION & API CLIENTS ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST"), "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"), "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 3306)),
}

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)
logger.info("LangChain ChatGroq client initialized.")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# --- 4. RAG & EMBEDDING SETUP ---
embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': embedding_device},
    encode_kwargs={'normalize_embeddings': False}
)
logger.info(f"HuggingFace Embeddings model loaded on device: {embedding_device}.")

pc = Pinecone(api_key=PINECONE_API_KEY)
logger.info("Pinecone client initialized.")

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()
logger.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")

# --- THE DEFINITIVE, HIGH-PRECISION PROMPT FOR SQL GENERATION ---
RAG_PROMPT_TEMPLATE = """
You are a world-class MySQL database analyst. Your ONLY function is to write a single, precise, and executable `SELECT` query to answer the user's question based on the provided database schema.

**CRITICAL DIRECTIVES:**
1.  **QUERY ONLY:** Your output MUST be a single SQL `SELECT` query, and nothing else. Do not provide explanations, comments, or any conversational text.
2.  **STRICT SCHEMA ADHERENCE:** You MUST only use the tables and columns explicitly listed in the schema context below. NEVER invent column names. For example, if the user asks for "top selling products" and there is no 'sales' column, you MUST create the logic to calculate it from the available columns (e.g., `product_sales.units_sold`).
3.  **HANDLE JOINS AND CALCULATIONS:** If a user's question requires information from multiple tables (e.g., "user names for recent orders" or "total revenue per product"), you MUST generate the correct `JOIN` statements. To calculate revenue, you MUST multiply quantity and price from the relevant tables (e.g., `SUM(order_items.quantity * order_items.price)`).
4.  **FAILURE CONDITION:** If the user's question is impossible to answer with the given schema (e.g., it asks for a 'customer address' but no such column exists) or is not a question about data (e.g., "hello"), your output MUST be the single word: `INVALID`.

### Database Schema:
{context}

### User's Question:
{question}
"""
rag_prompt = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])

# --- 5. SPEECH-TO-TEXT MODEL SETUP ---
WHISPER_MODEL_SIZE = "base"
stt_device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"
logger.info(f"Loading faster-whisper model '{WHISPER_MODEL_SIZE}'...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=stt_device, compute_type=compute_type)
logger.info("Faster-Whisper model loaded successfully.")

# --- 6. PYDANTIC MODELS ---
class TextQuery(BaseModel): message: str
class ChatResponse(BaseModel):
    message: str
    transcript: Optional[str] = None
    sql_query: Optional[str] = None
    data: Optional[List[Dict]] = None
    error: Optional[str] = None

# --- 7. CORE HELPER FUNCTIONS ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected(): return conn
    except mysql.connector.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

def get_database_schema(conn) -> List[str]:
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        schema_docs = []
        for table in tables:
            doc = f"Table name: `{table}`. Columns: "
            cursor.execute(f"DESCRIBE `{table}`")
            columns = [f"`{col[0]}` (type: {col[1]})" for col in cursor.fetchall()]
            doc += ", ".join(columns) + "."
            schema_docs.append(doc)
        cursor.close()
        return schema_docs
    except mysql.connector.Error as e:
        logger.error(f"Failed to fetch database schema: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch schema: {e}")

def generate_llm_response(user_message: str) -> str:
    try:
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt | llm | StrOutputParser()
        )
        response = rag_chain.invoke(user_message)
        return response.strip()
    except Exception as e:
        logger.error(f"RAG LLM response generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not generate a response from the LLM.")

def execute_sql_query(sql_query: str) -> List[Dict]:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql_query)
        return cursor.fetchall()
    except mysql.connector.Error as e:
        logger.error(f"SQL execution error for query '{sql_query}': {e}")
        raise HTTPException(status_code=400, detail=f"SQL Error: {e.msg}")
    finally:
        if conn and conn.is_connected(): conn.close()

async def transcribe_audio_with_pydub(audio_file: UploadFile) -> str:
    """Robustly transcribes any audio file format using pydub."""
    try:
        file_content = await audio_file.read()
        logger.info("Loading audio with pydub (requires ffmpeg)...")
        audio_segment = AudioSegment.from_file(io.BytesIO(file_content))
        
        logger.info("Converting audio to Whisper format (16kHz, mono)...")
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        samples = np.array(audio_segment.get_array_of_samples())
        audio_float32 = samples.astype(np.float32) / 32768.0
        
        logger.info("Applying noise reduction...")
        reduced_noise_audio = nr.reduce_noise(y=audio_float32, sr=16000)
        
        logger.info("Starting audio transcription...")
        segments, _ = whisper_model.transcribe(reduced_noise_audio, beam_size=5)
        transcript = "".join(segment.text for segment in segments)
        logger.info("Transcription finished.")
        return transcript.strip()
    except Exception as e:
        logger.error(f"Audio transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not transcribe audio. Ensure FFmpeg is correctly installed and in your system's PATH.")

# --- 8. API ENDPOINTS (FINAL LOGIC) ---
async def process_query(user_message: str) -> dict:
    try:
        logger.info(f"Processing user message: '{user_message}'")
        llm_response = generate_llm_response(user_message)

        if llm_response.strip().upper().startswith("SELECT"):
            sql_query = llm_response
            if not sql_query.endswith(';'):
                sql_query += ';'
            
            logger.info(f"LLM generated SQL query: {sql_query}")
            results = execute_sql_query(sql_query)
            message = f"Successfully executed query. Found {len(results)} result(s)."
            return {"message": message, "sql_query": sql_query, "data": results}
        elif "INVALID" in llm_response.upper():
             logger.info("LLM classified the query as invalid or conversational.")
             return {"message": "I can only answer questions that can be resolved with a SQL query. Please ask something specific about your data.", "error": "Invalid question."}
        else:
            logger.warning(f"LLM returned an unexpected response: '{llm_response}'")
            return {"message": "The AI returned an unexpected response that was not a valid SQL query.", "error": "Invalid AI response."}

    except HTTPException as he:
        # Catches errors from our own code (e.g., SQL execution error)
        return {"message": f"An error occurred: {he.detail}", "error": he.detail}
    except Exception as e:
        # Catches any other unexpected server-side error
        logger.error(f"Unexpected error in process_query: {e}", exc_info=True)
        return {"message": "An unexpected server error occurred.", "error": str(e)}

@app.post("/query-text", response_model=ChatResponse, summary="Query with Text")
async def query_text_endpoint(query: TextQuery):
    result = await process_query(query.message)
    return ChatResponse(**result)

@app.post("/query-audio", response_model=ChatResponse, summary="Query with Audio")
async def query_audio_endpoint(audio: UploadFile = File(...)):
    transcript = await transcribe_audio_with_pydub(audio)
    if not transcript.strip():
        return ChatResponse(message="No speech detected in audio.", transcript=transcript, error="No speech detected.")
    
    result = await process_query(transcript)
    return ChatResponse(transcript=transcript, **result)

@app.post("/index-schema", summary="Index Database Schema")
async def index_schema_endpoint():
    conn = None
    try:
        conn = get_db_connection()
        schema_documents = get_database_schema(conn)
        vectorstore.add_texts(texts=schema_documents)
        logger.info(f"Successfully indexed {len(schema_documents)} table schemas.")
        return {"message": "Database schema has been indexed successfully."}
    except Exception as e:
        logger.error(f"Schema indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn and conn.is_connected(): conn.close()

@app.get("/health", summary="Health Check")
async def health_check():
    db_status = "disconnected"
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            db_status = "connected"
    except Exception: pass
    return {"status": "healthy", "database": db_status}