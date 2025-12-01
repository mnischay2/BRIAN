import os
import json
import time
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import scripts.postgres_config as cfg
from scripts.lx_examples import Ex  

# --- Docling & LangExtract ---
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
import langextract as lx

OLLAMA_URL = ("http://localhost:11434") 
LLM_MODEL_ID = ( "llama3")
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
HISTORY_FILE = "ingestion_history.json"

print(f"⚡ Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda') 

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=cfg.TARGET_DB_NAME, 
            **cfg.PG_CREDENTIALS
        )
        register_vector(conn)
        return conn
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return None

# --- History Tracking Functions ---
def load_history():
    """Loads the ingestion history from JSON file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️  History file corrupted, starting fresh.")
            return {}
    return {}

def save_history(history):
    """Saves the ingestion history to JSON file."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"⚠️  Failed to save ingestion history: {e}")

def get_file_metadata(file_path):
    """Returns a dict of file metadata for verification."""
    stats = os.stat(file_path)
    return {
        "size": stats.st_size,
        "mtime": stats.st_mtime,
        "extension": os.path.splitext(file_path)[1].lower()
    }

# --- Core Logic ---

def run_docling(file_path: str):
    """
    Parses 2-column PDFs using Docling's Layout Analysis.
    Returns chunks respecting Reading Order.
    """
    print(f"  📄 [Docling] Analyzing Layout of {os.path.basename(file_path)}...")
    
    # 1. Configure Pipeline
    pipeline_opts = PdfPipelineOptions(do_table_structure=True)
    pipeline_opts.table_structure_options.mode = TableFormerMode.ACCURATE
    
    # 2. Correctly wrap the pipeline options
    pdf_format_options = PdfFormatOption(pipeline_options=pipeline_opts)
    
    # 3. Initialize converter
    converter = DocumentConverter(
        format_options={InputFormat.PDF: pdf_format_options} 
    )
    
    # 4. Run conversion
    print("    > Converting PDF document...")
    result = converter.convert(file_path)
    doc = result.document
    print("    > PDF Converted.")
    
    # 5. Smart Chunking
    chunks = []
    current_chunk_text = []
    current_page = 1
    
    for item in doc.texts:
        text = item.text.strip()
        if not text: 
            continue

        page = item.prov[0].page_no if item.prov else current_page
        
        # Heuristic: Break chunks on headers or size limit
        is_header = "header" in item.label.lower()
        
        if is_header and current_chunk_text:
            full_text = "\n".join(current_chunk_text)
            if len(full_text) > 50: 
                chunks.append({"page": current_page, "content": full_text})
            current_chunk_text = []
            current_page = page

        current_chunk_text.append(text)
        current_page = page
        
        # Max chunk size (approx chars)
        if sum(len(t) for t in current_chunk_text) > 1000:
            full_text = "\n".join(current_chunk_text)
            chunks.append({"page": current_page, "content": full_text})
            current_chunk_text = []

    # Add remaining text
    if current_chunk_text:
        full_text = "\n".join(current_chunk_text)
        if len(full_text) > 50:
            chunks.append({"page": current_page, "content": full_text})
    
    print(f"    > Extracted {len(chunks)} logical sections.")
    return chunks

def run_langextract(text_chunk: str):
    """Extracts strict metadata using Llama 3 via LangExtract."""
    print(f"    🤖 [LangExtract] Processing section...")
    
    prompt = """
    Extract key entities relevant to the document type:
    - Resumes: Skills, Experience, Education.
    - Textbooks: Topics, Definitions, Dates, Figures.
    - Stories: Characters, Locations, Events.
    - Research: Methods, Metrics, Findings.
    Return JSON.
    """
    
    examples = Ex
    
    try:
        result = lx.extract(
            text_or_documents=text_chunk,
            prompt_description=prompt,
            examples=examples,
            model_id=LLM_MODEL_ID,
            model_url=OLLAMA_URL, 
            temperature=0
        )
        
        data = {}
        for x in result.extractions:
            k, v = x.extraction_class, x.extraction_text
            if k not in data: data[k] = []
            data[k].append(v)
        return data
    except Exception as e:
        print(f"    [!] LangExtract Error: {e}")
        return {}

def ingest_file(file_path: str):
    """Parses, extracts metadata, and ingests a SINGLE file into DB."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    # 1. Parse PDF
    try:
        chunks = run_docling(file_path)
    except Exception as e:
        print(f"❌ Docling parsing failed for {os.path.basename(file_path)}: {e}")
        return False
    
    conn = get_db_connection()
    if not conn:
        print("❌ Could not connect to DB. Skipping file.")
        return False

    try:
        with conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                print(f"  > Chunk {i+1}/{len(chunks)} (Page {chunk['page']})")
                content = chunk["content"]
                
                # 2. Extract Metadata (LLM)
                meta_data = run_langextract(content)
                
                # 3. Generate Embedding
                embedding = embedding_model.encode(content).tolist()
                
                # 4. Insert into Postgres
                cur.execute(
                    """
                    INSERT INTO knowledge_base 
                    (source_file, page_number, raw_content, structured_data, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        os.path.basename(file_path),
                        chunk['page'],
                        content,
                        json.dumps(meta_data),
                        embedding 
                    )
                )
            conn.commit()
        
        print(f" Successfully ingested: {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        print(f" Error during ingestion of {os.path.basename(file_path)}: {e}")
        return False
    finally:
        conn.close()

def process_batch(folder_path: str):
    """Finds and ingests all PDFs in the given folder with history tracking."""
    if not os.path.exists(folder_path):
        print(f" Folder not found: {folder_path}")
        return

    history = load_history()

    files = [
        f for f in os.listdir(folder_path) 
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith('.pdf')
    ]

    if not files:
        print(f"  No .pdf files found in directory: {folder_path}")
        return

    print(f"\n Found {len(files)} PDFs in '{folder_path}'. checking history...\n")

    successful = 0
    failed = 0
    skipped = 0

    for idx, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)
        current_meta = get_file_metadata(file_path)
        
        if filename in history:
            prev_meta = history[filename]
            if (prev_meta.get("size") == current_meta["size"] and 
                prev_meta.get("mtime") == current_meta["mtime"]):                
                print(f" [{idx+1}/{len(files)}] Skipping {filename} (Already Ingested & Verified)")
                skipped += 1
                continue
            else:
                 print(f" [{idx+1}/{len(files)}] Re-ingesting {filename} (File changed)")
        else:
             print(f"🔹 [{idx+1}/{len(files)}] Processing: {filename}")

        if ingest_file(file_path):
            successful += 1
            history[filename] = current_meta
            save_history(history)
        else:
            failed += 1
        print("-" * 50)

    print("\n--- Batch Processing Summary ---")
    print(f"✅ Successful: {successful}")
    print(f"⏩ Skipped:    {skipped}")
    print(f"❌ Failed:     {failed}")
    print("-------------------------------")


if __name__ == "__main__":
    print("--- Batch RAG Ingestion Pipeline ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir_name = "knowledge_base"
    default_path = os.path.join(script_dir, default_dir_name)
    target_folder = default_path
    target_folder = target_folder.replace('"', '')
    if target_folder == default_path and not os.path.exists(default_path):
        try:
            os.makedirs(default_path)
            print(f"[*] Created default directory at '{default_path}'. Place your PDFs here and run again.")
        except OSError as e:
            print(f"Error creating directory: {e}")
    else:
        process_batch(target_folder)