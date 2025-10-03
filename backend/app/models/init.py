from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class AudioTranscriptionRequest(BaseModel):
    session_id: str
    role: str = "user"

class AudioTranscriptionResponse(BaseModel):
    text: str
    confidence: float
    session_id: str

class QueryRequest(BaseModel):
    session_id: str
    role: str = "user"
    text: str

class QueryResponse(BaseModel):
    result_table: List[Dict[str, Any]]
    summary: str
    session_id: str

class SessionMessage(BaseModel):
    role: str
    type: str
    content: str
    timestamp: datetime
    confidence: Optional[float] = None
    results: Optional[List[Dict[str, Any]]] = None

class AuditLogEntry(BaseModel):
    timestamp: datetime
    user: str
    session_id: str
    query_text: str
    query_spec: Dict[str, Any]
    executed_query: str
    results_count: int
    query_hash: str