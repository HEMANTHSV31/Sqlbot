from typing import List, Dict, Any
from sqlalchemy import create_engine, inspect, MetaData, Table
from pymongo import MongoClient
import hashlib
from app.config import DBType, settings

class SchemaIntrospector:
    def __init__(self, db_type: DBType):
        self.db_type = db_type
        self.engine = self._create_engine()
    
    def _create_engine(self):
        """Create database engine based on type"""
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
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def get_schema_hash(self) -> str:
        """Generate hash of current schema for change detection"""
        schema_info = self.introspect_schema()
        schema_str = str(schema_info)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def introspect_schema(self) -> List[Dict[str, Any]]:
        """Introspect database schema and return structured documentation"""
        if self.db_type == DBType.MONGODB:
            return self._introspect_mongodb()
        else:
            return self._introspect_sql()
    
    def _introspect_sql(self) -> List[Dict[str, Any]]:
        """Introspect SQL database schema"""
        inspector = inspect(self.engine)
        schema_docs = []
        
        for table_name in inspector.get_table_names():
            # Get columns
            columns = inspector.get_columns(table_name)
            column_info = [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"],
                    "default": col.get("default")
                }
                for col in columns
            ]
            
            # Get foreign keys
            foreign_keys = inspector.get_foreign_keys(table_name)
            fk_info = [
                {
                    "constrained_columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"]
                }
                for fk in foreign_keys
            ]
            
            # Create schema document
            doc = {
                "content": f"Table: {table_name}\nColumns: {[col['name'] for col in columns]}\nForeign Keys: {len(foreign_keys)}",
                "metadata": {
                    "db_type": self.db_type.value,
                    "table": table_name,
                    "columns": column_info,
                    "foreign_keys": fk_info,
                    "schema_version": self.get_schema_hash()
                }
            }
            schema_docs.append(doc)
        
        return schema_docs
    
    def _introspect_mongodb(self) -> List[Dict[str, Any]]:
        """Introspect MongoDB collections"""
        client = MongoClient(
            f"mongodb://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/"
        )
        db = client[settings.DB_NAME]
        schema_docs = []
        
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            
            # Sample documents to infer schema
            sample_docs = list(collection.find().limit(10))
            field_types = {}
            
            for doc in sample_docs:
                for key, value in doc.items():
                    field_types[key] = type(value).__name__
            
            doc = {
                "content": f"Collection: {collection_name}\nFields: {list(field_types.keys())}",
                "metadata": {
                    "db_type": self.db_type.value,
                    "table": collection_name,
                    "fields": field_types,
                    "schema_version": self.get_schema_hash()
                }
            }
            schema_docs.append(doc)
        
        return schema_docs