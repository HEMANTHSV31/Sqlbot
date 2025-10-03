import hashlib
from typing import List, Dict

import pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.config import settings


class PineconeClient:
    def __init__(self):
        pinecone.init(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.index_name = settings.PINECONE_INDEX_NAME
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=384,  # matches all-MiniLM-L6-v2
                metric="cosine"
            )
    
    def store_schema_embeddings(self, schema_docs: List[Dict], db_type: str):
        """Store schema documents as embeddings in Pinecone"""
        documents = [
            Document(page_content=doc["content"], metadata=doc["metadata"])
            for doc in schema_docs
        ]
        
        # Clear existing embeddings for this db_type
        self.delete_schema_embeddings(db_type)
        
        # Store new embeddings
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        return vectorstore
    
    def search_similar_schemas(self, query: str, db_type: str, k: int = 5):
        """Search for similar schema documents"""
        vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        results = vectorstore.similarity_search(
            query, 
            k=k,
            filter={"db_type": db_type}
        )
        return results
    
    def delete_schema_embeddings(self, db_type: str):
        """Delete all embeddings for a specific database type"""
        index = pinecone.Index(self.index_name)
        
        try:
            index.delete(filter={"db_type": db_type})
        except Exception as e:
            print(f"Error deleting embeddings: {e}")
