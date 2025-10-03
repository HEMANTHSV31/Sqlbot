import os
import time
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mysql.connector
from groq import Groq
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
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'chatbot_db'),
    'port': os.getenv('DB_PORT', '3306')
}

# Initialize clients
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Available Groq models - using current models
GROQ_MODEL = "llama-3.1-8b-instant"  # or "mixtral-8x7b-32768", "llama-3.1-70b-versatile"

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
1. Generate ONLY SELECT queries - no INSERT, UPDATE, DELETE, DROP, etc.
2. Only use the tables listed above.
3. If the request requires data outside these tables, return this exact fallback query:
   SELECT 'Invalid request. Only users, products, orders, order_items, and product_sales are supported.' AS error;
4. Prevent SQL injection - never use direct user input in queries.
5. If no table is explicitly mentioned, infer the best table based on context.
6. Always use proper JOINs when needed.
7. Include relevant WHERE clauses for filtering.
8. Use LIMIT for large result sets when appropriate.
9. Return ONLY the SQL query, nothing else.

Example conversions:
- "show me all users" → "SELECT * FROM users LIMIT 100;"
- "top 5 products by revenue" → "SELECT p.name, SUM(ps.revenue) as total_revenue FROM products p JOIN product_sales ps ON p.id = ps.product_id GROUP BY p.id, p.name ORDER BY total_revenue DESC LIMIT 5;"
- "orders from today" → "SELECT * FROM orders WHERE DATE(created_at) = CURDATE();"
"""

def get_db_connection():
    """Get database connection"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def generate_sql_query(user_message: str) -> str:
    """Generate SQL query using Groq LLM"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SQL_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=500
        )
        
        sql_query = chat_completion.choices[0].message.content.strip()
        
        # Clean up the SQL query - remove any markdown code blocks
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()
        
        logger.info(f"Generated SQL: {sql_query}")
        return sql_query
        
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail="Could not generate SQL query")

def execute_sql_query(sql_query: str) -> List[Dict]:
    """Execute SQL query and return results"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        start_time = time.time()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        execution_time = time.time() - start_time
        
        logger.info(f"SQL executed in {execution_time:.2f}s, returned {len(results)} rows")
        
        return results
        
    except mysql.connector.Error as e:
        logger.error(f"SQL execution error: {e}")
        raise HTTPException(status_code=500, detail=f"SQL execution error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during SQL execution: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if conn and conn.is_connected():
            conn.close()

async def transcribe_audio(audio_file: UploadFile) -> str:
    """Transcribe audio using Whisper API"""
    try:
        # Read file content - use await since this is an async function
        file_content = await audio_file.read()
        
        # Use OpenAI Whisper API
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", file_content, "audio/wav")
        )
        
        logger.info(f"Transcribed audio: {transcript.text}")
        return transcript.text
        
    except Exception as e:
        logger.error(f"Whisper API error: {e}")
        raise HTTPException(status_code=500, detail="Could not transcribe audio")

def refine_transcription(transcript: str) -> str:
    """Refine transcription using GPT for grammar correction"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Fix any grammar or spelling errors in the following text while preserving the original meaning. Return only the corrected text."},
                {"role": "user", "content": transcript}
            ],
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=500
        )
        
        refined_text = chat_completion.choices[0].message.content.strip()
        logger.info(f"Refined transcription: {refined_text}")
        return refined_text
        
    except Exception as e:
        logger.error(f"Transcription refinement error: {e}")
        return transcript  # Return original if refinement fails

@app.post("/query-text", response_model=ChatResponse)
async def query_text(query: TextQuery):
    """Handle text queries"""
    start_time = time.time()
    
    try:
        # Generate SQL query
        sql_query = generate_sql_query(query.message)
        
        # Execute SQL query
        results = execute_sql_query(sql_query)
        
        # Format response message
        if results and 'error' in results[0]:
            message = results[0]['error']
        elif not results:
            message = "No results found for your query."
        else:
            message = f"Found {len(results)} result(s) for your query."
        
        response_time = time.time() - start_time
        logger.info(f"Text query processed in {response_time:.2f}s")
        
        return ChatResponse(
            message=message,
            sql_query=sql_query,
            data=results
        )
        
    except HTTPException as he:
        logger.error(f"HTTP Exception in text query: {he.detail}")
        return ChatResponse(
            message=f"Error: {he.detail}",
            error=he.detail
        )
    except Exception as e:
        logger.error(f"Unexpected error in text query: {e}")
        return ChatResponse(
            message="An unexpected error occurred while processing your request.",
            error=str(e)
        )

@app.post("/query-audio", response_model=ChatResponse)
async def query_audio(audio: UploadFile = File(...)):
    """Handle audio queries"""
    start_time = time.time()
    
    try:
        # Validate audio file
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Transcribe audio
        transcript = await transcribe_audio(audio)  # Now using await since it's async
        
        if not transcript.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # Refine transcription
        refined_transcript = refine_transcription(transcript)
        
        # Generate SQL query from refined transcript
        sql_query = generate_sql_query(refined_transcript)
        
        # Execute SQL query
        results = execute_sql_query(sql_query)
        
        # Format response message
        if results and 'error' in results[0]:
            message = results[0]['error']
        elif not results:
            message = f"No results found for: '{refined_transcript}'"
        else:
            message = f"Found {len(results)} result(s) for: '{refined_transcript}'"
        
        response_time = time.time() - start_time
        logger.info(f"Audio query processed in {response_time:.2f}s")
        
        return ChatResponse(
            message=message,
            sql_query=sql_query,
            data=results
        )
        
    except HTTPException as he:
        logger.error(f"HTTP Exception in audio query: {he.detail}")
        return ChatResponse(
            message=f"Error: {he.detail}",
            error=he.detail
        )
    except Exception as e:
        logger.error(f"Unexpected error in audio query: {e}")
        return ChatResponse(
            message="An unexpected error occurred while processing your audio request.",
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        conn.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

@app.get("/models")
async def get_available_models():
    """Get available Groq models"""
    try:
        models = groq_client.models.list()
        model_ids = [model.id for model in models.data]
        return {"available_models": model_ids}
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {"available_models": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)