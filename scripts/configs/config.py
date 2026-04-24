import logging
import os

# --- CONFIGURATION ---
CONF = { 
    # System Settings 
    "DEBUG": os.getenv("AETHER_DEBUG", "True").lower() == "true", # Default to True for debugging
    
    # LLM Settings
    "LLM_USAGE": os.getenv("LLM_USAGE", "local").lower(),  # "local" (Ollama) or "cloud" (Gemini)
    "LLM_MODEL": os.getenv("AETHER_MODEL", "llama3:latest"),  # Ollama model
    "LLM_BASE_URL": os.getenv("AETHER_LLM_URL", "http://localhost:11434"),
    "GEMINI_MODEL": "gemini-2.5-flash",  # Google Gemini Cloud model
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "AIzaSyB9WY0_v6zQWpouYIOVFOEHk0qRWRRa3CU"),  # Add to .env file
    "LLM_NUM_CTX" : 16834,
    "LLM_NUM_GPU" : 99,
    "USER_AGENT": "AetherIntel/1.0; +http://localhost)",
    
    "TIMEOUT_FETCH": 10,
    "MAX_CONTENT_CHARS": 15000, 
    "MIN_CONFIDENCE": 1, #CONIFDENCE ON TIER USAGE # not for DR
    
    "DEVICE": "cuda",
    "INGEST_BATCH_SIZE": 50,

     # RAG / Data Settings
    "EMBEDDING_MODEL_NAME": "sentence-transformers/all-mpnet-base-v2",
    "CHROMA_PERSIST_DIR": os.path.join(os.getcwd(), "chroma_db"),
    "COLLECTION_NAME": "knowledge_base",
    "CHUNK_SIZE": 2000,
    "CHUNK_OVERLAP": 300,

    # Deep Research specific
    "DR_Tools_budgets": 20,
    "DR_QUERIES_PER_PHASE": 3,
    "DR_WEB_RESULTS_PER_QUERY": 10,
    "DR_RAG_TOP_K": 10,
    
    #normal thresholds
    "max_tool_calls_per_question": 20,
    "MIN_TOOL_CALLS_BEFORE_ANSWER": 2,
    "WEB_RESULTS_per_query": 10,
    "RAG_TOP_K": 6,
    "RAG_PHASES": 2,
    "WEB_PHASES": 2,
    "RAG_SCORE_THRESHOLD": 0.7, 

    "PDF_SEGMENT_WORKERS": 2, 
    "PDF_OCR_WORKERS":     2,   

}

# Configure structured logging
log_level = logging.DEBUG if CONF["DEBUG"] else logging.INFO
logging.basicConfig(
    level=log_level, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def get_logger(name: str):
    return logging.getLogger(f"Aether.{name}")