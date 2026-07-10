import os
from typing import TypedDict, Annotated, Sequence, Any, Optional
import json
import re
import mysql.connector
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
import operator

# Define State for the Graph
class GraphState(TypedDict):
    query: str
    config: dict
    schema_context: str
    pinecone_context: str
    sql_query: str
    columns: list
    rows: list
    is_chart: bool
    chart_type: str
    error: str
    retries: int

def get_connection(config: dict):
    return mysql.connector.connect(
        host=config.get("db_host"),
        user=config.get("db_user"),
        password=config.get("db_password"),
        database=config.get("db_name")
    )

def fetch_schema(config: dict):
    conn = get_connection(config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM information_schema.columns
        WHERE table_schema = %s
    """, (config.get('db_name'),))
    schema = cursor.fetchall()
    cursor.close()
    conn.close()
    return schema

def build_table_context(schema):
    context = {}
    for row in schema:
        table = row.get('table_name') or row.get('TABLE_NAME')
        column = row.get('column_name') or row.get('COLUMN_NAME')
        if table and column:
            dtype = row.get('data_type') or row.get('DATA_TYPE', 'unknown')
            if table not in context:
                context[table] = []
            context[table].append(f"`{column}` ({dtype})")
    return context

def detect_relevant_tables(query, schema):
    query_lower = query.lower()
    relevant_tables = set()
    for row in schema:
        table = row.get('table_name') or row.get('TABLE_NAME')
        column = row.get('column_name') or row.get('COLUMN_NAME')
        if table and column:
            if table.lower() in query_lower or column.lower() in query_lower:
                relevant_tables.add(table)
    return list(relevant_tables)


# Node: Fetch Context
def fetch_context_node(state: GraphState):
    query = state["query"]
    config = state["config"]
    
    schema = fetch_schema(config)
    relevant_tables = detect_relevant_tables(query, schema)
    
    table_context_dict = build_table_context(schema)
    context_str = "\n".join(
        [f"Table {tbl}: {', '.join(cols)}" for tbl, cols in table_context_dict.items() if tbl in relevant_tables]
    )
    if not context_str:
         context_str = "\n".join(
             [f"Table {tbl}: {', '.join(cols)}" for tbl, cols in table_context_dict.items()]
         )

    rag_context = ""
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if pinecone_key:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            pc = Pinecone(api_key=pinecone_key)
            index_name = os.getenv("PINECONE_INDEX_NAME", "dynamic_schema_index")
            if index_name in [i["name"] for i in pc.list_indexes()]:
                pinecone_index = pc.Index(index_name)
                vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
                docs = vector_store.similarity_search(query, k=5)
                rag_context = "\n".join([d.page_content for d in docs])
        except Exception as e:
            print(f"Warning: Pinecone retrieval failed: {e}")

    return {"schema_context": context_str, "pinecone_context": rag_context, "retries": 0, "error": ""}

# Node: Generate SQL
def generate_sql_node(state: GraphState):
    query = state["query"]
    config = state["config"]
    schema_context = state["schema_context"]
    pinecone_context = state["pinecone_context"]
    error = state.get("error", "")

    prompt_template = """
You are an expert SQL data assistant. 
Given this database schema:
{schema_context}

And additional vector DB context:
{pinecone_context}

And the user question: "{query}"
"""
    if error:
        prompt_template += f"\nYour previous query resulted in this error: {error}. Please correct the SQL query to fix this error."
        
    prompt_template += """
Generate a single safe MySQL SELECT statement (no updates/inserts/deletes).
Do not include any explanation or markdown blocks. Just return the raw SQL query.
"""
    
    prompt = PromptTemplate(
        input_variables=["schema_context", "pinecone_context", "query"],
        template=prompt_template
    )

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Groq API Key is missing from backend environment variables.")

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=groq_api_key)
    chain = prompt | llm
    
    result = chain.invoke({
        "schema_context": schema_context,
        "pinecone_context": pinecone_context,
        "query": query
    })
    
    sql_query = result.content.strip()
    sql_query = re.sub(r"```sql|```", "", sql_query).strip()
    
    return {"sql_query": sql_query}


# Node: Execute SQL
def execute_sql_node(state: GraphState):
    sql_query = state["sql_query"]
    config = state["config"]
    retries = state.get("retries", 0)
    
    if not sql_query.lower().startswith("select"):
        return {"error": "Only SELECT queries are allowed", "retries": retries + 1}

    try:
        conn = get_connection(config)
        cur = conn.cursor()
        cur.execute(sql_query)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        data = [dict(zip(columns, row)) for row in rows]
        
        chartable = False
        if len(columns) >= 2:
            numeric_cols = [c for c in columns if any(x in c.lower() for x in ["amount", "marks", "count", "total", "score", "fees", "price"])]
            time_cols = [c for c in columns if any(x in c.lower() for x in ["date", "time", "month", "year", "name"])]
            if numeric_cols and time_cols:
                chartable = True
                
        return {
            "columns": columns,
            "rows": data,
            "error": "",
            "is_chart": chartable,
            "chart_type": "bar" if chartable else None
        }
    except Exception as e:
        return {"error": str(e), "retries": retries + 1}

# Conditional Edge
def should_continue(state: GraphState):
    if state.get("error") and state.get("retries", 0) < 3:
        return "generate_sql"
    return END

# Build the Graph
workflow = StateGraph(GraphState)
workflow.add_node("fetch_context", fetch_context_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)

workflow.set_entry_point("fetch_context")
workflow.add_edge("fetch_context", "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_conditional_edges("execute_sql", should_continue, {"generate_sql": "generate_sql", END: END})

app_graph = workflow.compile()
