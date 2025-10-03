# import sqlalchemy
# from sqlalchemy import create_engine, text
# from pymongo import MongoClient
# from typing import List, Dict, Any
# from app.config import DBType, settings

# class DatabaseConnection:
#     def __init__(self, db_type: DBType, read_only: bool = True):
#         self.db_type = db_type
#         self.read_only = read_only
#         self.connection = self._create_connection()
    
#     def _create_connection(self):
#         if self.db_type == DBType.SQLITE:
#             return create_engine(f"sqlite:///{settings.DB_NAME}")
#         elif self.db_type == DBType.POSTGRESQL:
#             return create_engine(
#                 f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
#             )
#         elif self.db_type == DBType.MYSQL:
#             return create_engine(
#                 f"mysql+mysqlconnector://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
#             )
#         elif self.db_type == DBType.MONGODB:
#             return MongoClient(
#                 f"mongodb://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/"
#             )[settings.DB_NAME]
#         else:
#             raise ValueError(f"Unsupported database type: {self.db_type}")
    
#     async def execute_query(self, query) -> List[Dict[str, Any]]:
#         """Execute query and return results"""
#         try:
#             if self.db_type == DBType.MONGODB:
#                 return await self._execute_mongo_query(query)
#             else:
#                 return await self._execute_sql_query(query)
#         except Exception as e:
#             raise Exception(f"Query execution failed: {str(e)}")
    
#     async def _execute_sql_query(self, query: str) -> List[Dict[str, Any]]:
#         """Execute SQL query"""
#         with self.connection.connect() as conn:
#             result = conn.execute(text(query))
#             columns = result.keys()
#             rows = result.fetchall()
#             return [dict(zip(columns, row)) for row in rows]
    
#     async def _execute_mongo_query(self, pipeline: List[Dict]) -> List[Dict[str, Any]]:
#         """Execute MongoDB aggregation pipeline"""
#         # For simplicity, execute on first collection found
#         collections = self.connection.list_collection_names()
#         if not collections:
#             return []
        
#         collection = self.connection[collections[0]]
#         cursor = collection.aggregate(pipeline)
#         return list(cursor)

# def get_database_connection(db_type: DBType, read_only: bool = True) -> DatabaseConnection:
#     return DatabaseConnection(db_type, read_only)


import sqlalchemy
from sqlalchemy import create_engine, text
from pymongo import MongoClient
from typing import List, Dict, Any
import asyncio
from app.config import DBType, settings

class DatabaseConnection:
    def __init__(self, db_type: DBType, read_only: bool = True):
        self.db_type = db_type
        self.read_only = read_only
        self.engine = self._create_engine()
    
    def _create_engine(self):
        if self.db_type == DBType.SQLITE:
            return create_engine(f"sqlite:///{settings.DB_NAME}")
        elif self.db_type == DBType.POSTGRESQL:
            return create_engine(
                f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            )
        elif self.db_type == DBType.MYSQL:
            return create_engine(
                f"mysql+mysqlconnector://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            )
        elif self.db_type == DBType.MONGODB:
            return MongoClient(
                f"mongodb://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/"
            )[settings.DB_NAME]
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    async def execute_query(self, query) -> List[Dict[str, Any]]:
        """Execute query and return results"""
        try:
            if self.db_type == DBType.MONGODB:
                return await self._execute_mongo_query(query)
            else:
                return await self._execute_sql_query(query)
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    
    async def _execute_sql_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_sql_sync, query)
    
    def _execute_sql_sync(self, query: str) -> List[Dict[str, Any]]:
        """Synchronous SQL execution"""
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]
    
    async def _execute_mongo_query(self, pipeline: List[Dict]) -> List[Dict[str, Any]]:
        """Execute MongoDB aggregation pipeline in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_mongo_sync, pipeline)
    
    def _execute_mongo_sync(self, pipeline: List[Dict]) -> List[Dict[str, Any]]:
        """Synchronous MongoDB execution"""
        collections = self.engine.list_collection_names()
        if not collections:
            return []
        
        collection = self.engine[collections[0]]
        cursor = collection.aggregate(pipeline)
        return list(cursor)

def get_database_connection(db_type: DBType, read_only: bool = True) -> DatabaseConnection:
    return DatabaseConnection(db_type, read_only)