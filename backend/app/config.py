import os
from typing import Optional
from pydantic_settings import BaseSettings
from enum import Enum

class DBType(str, Enum):
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"

class VectorDBType(str, Enum):
    PINECONE = "pinecone"
    FAISS = "faiss"
    MILVUS = "milvus"

class Settings(BaseSettings):
    # Groq API
    GROQ_API_KEY: str
    
    # Database Configuration
    DB_TYPE: DBType = DBType.SQLITE
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = None
    DB_NAME: str = "chatbot_db"
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    
    # Vector DB Configuration (Pinecone default)
    VECTOR_DB_TYPE: VectorDBType = VectorDBType.PINECONE
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str = "schema-embeddings"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Audio Processing
    WHISPER_MODEL: str = "base"
    MAX_AUDIO_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    class Config:
        env_file = ".env"

settings = Settings()