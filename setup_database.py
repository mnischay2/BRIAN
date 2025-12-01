import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import scripts.postgres_config as cfg

def get_sys_connection():
    return psycopg2.connect(dbname="postgres", **cfg.PG_CREDENTIALS)

def get_target_connection():
    return psycopg2.connect(dbname=cfg.TARGET_DB_NAME, **cfg.PG_CREDENTIALS)

def create_database_if_missing():
    conn = None
    try:
        conn = get_sys_connection()
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (cfg.TARGET_DB_NAME,))
        exists = cur.fetchone()
        
        if not exists:
            print(f"[*] Database '{cfg.TARGET_DB_NAME}' not found. Creating...")
            cur.execute(f'CREATE DATABASE "{cfg.TARGET_DB_NAME}"')
            print(f"[+] Database '{cfg.TARGET_DB_NAME}' created successfully.")
        else:
            print(f"[*] Database '{cfg.TARGET_DB_NAME}' already exists.")
            
        cur.close()
    except Exception as e:
        print(f"[!] Error creating database: {e}")
    finally:
        if conn: conn.close()

def setup_schema():
    conn = None
    try:
        print(f"[*] Connecting to {cfg.TARGET_DB_NAME} to setup schema...")
        conn = get_target_connection()
        conn.autocommit = True 
        cur = conn.cursor()

        print("Enabling pgvector extension...")
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            print(f"FAILED to enable pgvector: {e}")
            return

        print("Creating 'sessions' table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            question_number INTEGER,
            date DATE,
            time TIME,
            session_id VARCHAR(50),
            question TEXT,
            response TEXT
        );
        """)

        print(f"Creating 'knowledge_base' table (Dim: {cfg.EMBEDDING_DIM})...")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id SERIAL PRIMARY KEY,
            source_file TEXT NOT NULL,
            page_number INT,
            raw_content TEXT,
            structured_data JSONB,
            embedding VECTOR({cfg.EMBEDDING_DIM}),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        print("Creating vector index (HNSW)...")
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_hnsw_embedding
        ON knowledge_base
        USING hnsw (embedding vector_l2_ops);
        """)

        print("Creating metadata index (GIN)...")
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_gin_structured_data
        ON knowledge_base
        USING GIN (structured_data);
        """)

        print("Database setup complete.")
        cur.close()

    except Exception as e:
        print(f"Schema Setup Failed: {e}")
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    create_database_if_missing()
    setup_schema()