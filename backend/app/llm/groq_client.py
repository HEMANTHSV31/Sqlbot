import groq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
import json
from typing import List, Dict, Any
from app.config import settings

class JSONOutputParser(BaseOutputParser):
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse JSON output from LLM"""
        try:
            # Extract JSON from text if it's wrapped in markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

class GroqClient:
    def __init__(self):
        self.client = groq.Groq(api_key=settings.GROQ_API_KEY)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        # Prompt template for generating query specs
        self.query_prompt = PromptTemplate(
            template="""You are a SQL expert. Convert the natural language query into a structured JSON query specification.

Database Schema Context:
{schema_context}

User Query: {query}

Previous Conversation:
{conversation_history}

Output only valid JSON following this exact schema:
{{
    "type": "select",
    "table": "main_table_name",
    "columns": [
        {{"name": "column1", "alias": "optional_alias"}},
        {{"name": "column2"}}
    ],
    "where": "condition if any",
    "group_by": ["column1"],
    "order_by": ["column1 DESC"],
    "limit": 10,
    "joins": [
        {{"table": "other_table", "on": "main_table.id = other_table.main_id", "type": "inner"}}
    ]
}}

JSON Output:""",
            input_variables=["schema_context", "query", "conversation_history"]
        )
    
    def generate_query_spec(self, query: str, schema_context: str, conversation_history: str = "") -> Dict[str, Any]:
        """Generate structured query spec from natural language"""
        prompt = self.query_prompt.format(
            schema_context=schema_context,
            query=query,
            conversation_history=conversation_history or "No previous conversation"
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content
        parser = JSONOutputParser()
        return parser.parse(content)
    
    def generate_summary(self, query: str, results: List[Dict]) -> str:
        """Generate natural language summary of query results"""
        results_str = json.dumps(results[:5], indent=2)  # Sample first 5 rows
        
        prompt = f"""Summarize the following database query results in a helpful, natural language response.

User Question: {query}

Query Results (first 5 rows):
{results_str}

Provide a concise summary that answers the user's question based on the data. If no results were found, suggest alternative approaches.

Summary:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content