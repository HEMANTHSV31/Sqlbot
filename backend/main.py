import os
import time
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from groq import Groq, GroqError
import openai
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "chatbot_db"),
    "port": int(os.getenv("DB_PORT", 3306)),  # Convert to int
}

# Initialize Groq client safely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment variables")
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment variables")
openai.api_key = OPENAI_API_KEY

# Groq model
GROQ_MODEL = "llama-3.1-8b-instant"

# Pydantic models
class TextQuery(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    sql_query: Optional[str] = None
    data: Optional[List[Dict]] = None
    error: Optional[str] = None

# System prompt for SQL generation
SQL_SYSTEM_PROMPT = """
You are a SQL query generator. Your task is to generate ONLY SELECT queries for the MySQL database.

Available tables:
- users(id, name, email, created_at)
- products(id, name, sku, price, created_at)
- orders(id, user_id, total, created_at)
- order_items(id, order_id, product_id, quantity, price)
- product_sales(id, product_id, sale_date, units_sold, revenue)

Rules:
1. Generate ONLY SELECT queries.
2. Only use the tables listed above.
3. If the request requires data outside these tables, return this exact fallback query:
   SELECT 'Invalid request. Only users, products, orders, order_items, and product_sales are supported.' AS error;
4. Prevent SQL injection - never use direct user input in queries.
5. If no table is explicitly mentioned, infer the best table based on context.
6. Always use proper JOINs when needed.
7. Include relevant WHERE clauses for filtering.
8. Use LIMIT for large result sets when appropriate.
9. Return ONLY the SQL query, nothing else.
"""

# Database connection
def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

# Generate SQL query using Groq
def generate_sql_query(user_message: str) -> str:
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SQL_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=500
        )
        sql_query = completion.choices[0].message.content.strip()
        # Remove markdown code block
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        return sql_query.strip()
    except GroqError as e:
        logger.error(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail="Could not generate SQL query")

# Execute SQL query
def execute_sql_query(sql_query: str) -> List[Dict]:
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql_query)
        results = cursor.fetchall()
        return results
    except mysql.connector.Error as e:
        logger.error(f"SQL execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn and conn.is_connected():
            conn.close()

# Transcribe audio
async def transcribe_audio(audio_file: UploadFile) -> str:
    try:
        file_content = await audio_file.read()
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", file_content, "audio/wav")
        )
        return transcript.text
    except Exception as e:
        logger.error(f"Whisper API error: {e}")
        raise HTTPException(status_code=500, detail="Could not transcribe audio")

# Refine transcription
def refine_transcription(transcript: str) -> str:
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Fix grammar and spelling errors while preserving meaning."},
                {"role": "user", "content": transcript}
            ],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=500
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Transcription refinement error: {e}")
        return transcript

# Routes
@app.post("/query-text", response_model=ChatResponse)
async def query_text(query: TextQuery):
    try:
        sql_query = generate_sql_query(query.message)
        results = execute_sql_query(sql_query)
        message = results[0]["error"] if results and "error" in results[0] else f"Found {len(results)} result(s)."
        return ChatResponse(message=message, sql_query=sql_query, data=results)
    except HTTPException as he:
        return ChatResponse(message=f"Error: {he.detail}", error=he.detail)

@app.post("/query-audio", response_model=ChatResponse)
async def query_audio(audio: UploadFile = File(...)):
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    transcript = await transcribe_audio(audio)
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="No speech detected in audio")
    refined_transcript = refine_transcription(transcript)
    sql_query = generate_sql_query(refined_transcript)
    results = execute_sql_query(sql_query)
    message = results[0]["error"] if results and "error" in results[0] else f"Found {len(results)} result(s)."
    return ChatResponse(message=message, sql_query=sql_query, data=results)

@app.get("/health")
async def health_check():
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

@app.get("/models")
async def get_available_models():
    try:
        models = groq_client.models.list()
        return {"available_models": [m.id for m in models.data]}
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {"available_models": [], "error": str(e)}