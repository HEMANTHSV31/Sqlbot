from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError
import json

class ColumnSpec(BaseModel):
    name: str
    alias: str = None

class JoinSpec(BaseModel):
    table: str
    on: str
    type: str = "inner"  # inner, left, right

class QuerySpec(BaseModel):
    type: str = "select"
    table: str
    columns: List[ColumnSpec] = []
    where: str = None
    group_by: List[str] = []
    having: str = None
    order_by: List[str] = []
    limit: int = None
    joins: List[JoinSpec] = []

class QueryValidator:
    @staticmethod
    def validate_spec(spec_dict: Dict[str, Any]) -> QuerySpec:
        """Validate the query specification against the schema"""
        try:
            return QuerySpec(**spec_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid query spec: {e}")
    
    @staticmethod
    def is_safe_query(spec: QuerySpec) -> bool:
        """Ensure query is read-only and safe"""
        # Only SELECT queries allowed
        if spec.type.lower() != "select":
            return False
        
        # Check for dangerous patterns
        dangerous_keywords = ["insert", "update", "delete", "drop", "alter", "create"]
        query_str = json.dumps(spec.dict()).lower()
        
        for keyword in dangerous_keywords:
            if keyword in query_str:
                return False
        
        return True