import os
import psycopg
import ollama
from pgvector.psycopg import Vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "llama3")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

print("⚡ Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg.connect(DATABASE_URL)
    from pgvector.psycopg import register_vector
    register_vector(conn)
    return conn

def hybrid_search(query: str, topic_filter: str = None, limit: int = 5):
    """
    Searches knowledge base using Vectors + Structured Metadata.
    """
    print(f"  > Searching for query: '{query}'")
    if topic_filter:
        print(f"  > Filtering by topic: '{topic_filter}'")

    query_vec = Vector(embedding_model.encode(query))
    
    sql = "SELECT raw_content FROM knowledge_base"
    params = {"emb": query_vec, "limit": limit}
    conditions = []

    # Filter by the "Topic" metadata key we defined in ingest.py
    if topic_filter:
        # This syntax checks if the JSONB array 'Topic' contains the filter value
        conditions.append("structured_data->'Topic' ? %(topic)s")
        params['topic'] = topic_filter

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    # Sort by vector distance (nearest neighbors)
    sql += " ORDER BY embedding <-> %(emb)s LIMIT %(limit)s"
    
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        results = [row[0] for row in cur.fetchall()]
    conn.close()
    
    print(f"  > Found {len(results)} relevant chunks.")
    return results

