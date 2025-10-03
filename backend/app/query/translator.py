from typing import Dict, Any, List
from app.config import DBType
from app.query.spec_validator import QuerySpec
import re

class QueryTranslator:
    def __init__(self, db_type: DBType):
        self.db_type = db_type
    
    def spec_to_sql(self, spec: QuerySpec) -> str:
        """Convert query spec to parameterized SQL"""
        columns = self._format_columns(spec.columns)
        from_clause = self._format_from(spec.table)
        where_clause = self._format_where(spec.where) if spec.where else ""
        group_clause = self._format_group_by(spec.group_by) if spec.group_by else ""
        order_clause = self._format_order_by(spec.order_by) if spec.order_by else ""
        limit_clause = f"LIMIT {spec.limit}" if spec.limit else ""
        join_clause = self._format_joins(spec.joins) if spec.joins else ""
        
        sql = f"SELECT {columns} FROM {from_clause}"
        if join_clause:
            sql += f" {join_clause}"
        if where_clause:
            sql += f" {where_clause}"
        if group_clause:
            sql += f" {group_clause}"
        if order_clause:
            sql += f" {order_clause}"
        if limit_clause:
            sql += f" {limit_clause}"
        
        return sql
    
    def _format_columns(self, columns: List[Dict]) -> str:
        if not columns:
            return "*"
        formatted_columns = []
        for col in columns:
            if self.db_type == DBType.MYSQL:
                col_str = f"`{col['name']}`"
            else:
                col_str = f'"{col["name"]}"'
            if col.get("alias"):
                col_str += f' AS "{col["alias"]}"'
            formatted_columns.append(col_str)
        return ", ".join(formatted_columns)
    
    def _format_from(self, table: str) -> str:
        if self.db_type == DBType.MYSQL:
            return f"`{table}`"
        else:
            return f'"{table}"'
    
    def _format_where(self, where: str) -> str:
        # Basic sanitization - in production, use proper parameterization
        where = re.sub(r';', '', where)  # Remove semicolons
        where = re.sub(r'--', '', where)  # Remove SQL comments
        return f"WHERE {where}"
    
    def _format_group_by(self, group_by: List[str]) -> str:
        if self.db_type == DBType.MYSQL:
            columns = ", ".join(f"`{col}`" for col in group_by)
        else:
            columns = ", ".join(f'"{col}"' for col in group_by)
        return f"GROUP BY {columns}"
    
    def _format_order_by(self, order_by: List[str]) -> str:
        return f"ORDER BY {', '.join(order_by)}"
    
    def _format_joins(self, joins: List[Dict]) -> str:
        join_clauses = []
        for join in joins:
            join_type = join.get("type", "inner").upper()
            if self.db_type == DBType.MYSQL:
                table_ref = f"`{join['table']}`"
            else:
                table_ref = f'"{join["table"]}"'
            join_clauses.append(f"{join_type} JOIN {table_ref} ON {join['on']}")
        return " ".join(join_clauses)

class MongoQueryTranslator:
    def spec_to_mongo(self, spec: QuerySpec) -> List[Dict]:
        """Convert query spec to MongoDB aggregation pipeline"""
        pipeline = []
        
        # $match stage for WHERE clause
        if spec.where:
            match_stage = self._parse_where(spec.where)
            if match_stage:
                pipeline.append({"$match": match_stage})
        
        # $project stage for columns
        if spec.columns:
            projection = {col["name"]: 1 for col in spec.columns}
            projection["_id"] = 0  # Exclude MongoDB _id by default
            pipeline.append({"$project": projection})
        
        # $group stage
        if spec.group_by:
            group_id = {field: f"${field}" for field in spec.group_by}
            group_stage = {"_id": group_id}
            
            # Include all fields in the group
            for field in spec.group_by:
                group_stage[field] = {"$first": f"${field}"}
            
            pipeline.append({"$group": group_stage})
        
        # $sort stage
        if spec.order_by:
            sort_stage = {}
            for field in spec.order_by:
                if " DESC" in field.upper():
                    field_name = field.upper().replace(" DESC", "").strip()
                    sort_stage[field_name] = -1
                else:
                    field_name = field.upper().replace(" ASC", "").strip()
                    sort_stage[field_name] = 1
            pipeline.append({"$sort": sort_stage})
        
        # $limit stage
        if spec.limit:
            pipeline.append({"$limit": spec.limit})
        
        return pipeline
    
    def _parse_where(self, where: str) -> Dict:
        """Simple WHERE clause parser for MongoDB"""
        # Basic equality conditions
        conditions = {}
        if "=" in where:
            parts = where.split("=")
            if len(parts) == 2:
                field = parts[0].strip()
                value = parts[1].strip().strip("'\"")
                conditions[field] = value
        return conditions