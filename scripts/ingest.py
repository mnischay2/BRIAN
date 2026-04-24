import os

import hashlib
import sqlite3
import argparse
import json
import asyncio
import ollama  # Native Ollama library
from datetime import datetime
from typing import List

# Internal Aether Imports
try:
    from scripts.chat_llm import chat_snt_off, async_chat_snt_off
    from scripts.configs.config import CONF
    from scripts.logger import setup_aether_logger, Colors
    from scripts.silence_chatter import silence
    from scripts.pdf_reader import AdvancedPDFProcessor
except ImportError:
    from pdf_reader import AdvancedPDFProcessor
    from configs.config import CONF
    from logger import setup_aether_logger, Colors
    from silence_chatter import silence

silence()  # Mute noisy libraries globally before anything else imports them

# LangChain & Vector DB
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = setup_aether_logger("Ingest")

class DocumentRegistry:
    def __init__(self, db_path="aether_registry.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS doc_registry (
                    doc_id TEXT PRIMARY KEY,
                    doc_name TEXT,
                    doc_path TEXT,
                    doc_type TEXT,
                    doc_category TEXT,
                    keywords TEXT,
                    file_hash TEXT UNIQUE,
                    date_ingested DATETIME,
                    chunk_count INTEGER
                )
            """)
            conn.commit()

    def is_duplicate(self, file_hash: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM doc_registry WHERE file_hash = ?", (file_hash,))
            return cursor.fetchone() is not None

    def register_doc(self, metadata: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO doc_registry 
                (doc_id, doc_name, doc_path, doc_type, doc_category, keywords, file_hash, date_ingested, chunk_count)
                VALUES (:doc_id, :doc_name, :doc_path, :doc_type, :doc_category, :keywords, :file_hash, :date_ingested, :chunk_count)
            """, metadata)
            conn.commit()

class DocumentProcessor:
    def __init__(self, mode: str = "advanced"):
        self.mode = mode
        self.registry = DocumentRegistry()
        self.default_folder = os.path.join(os.getcwd(), "knowledge_files")
        self._ensure_knowledge_dir()
        
        # Ollama configuration from CONF
        self.model = CONF.get("LLM_MODEL", "qwen2.5:7b")

        self.pdf_engine = AdvancedPDFProcessor(
            chunk_size=CONF.get("CHUNK_SIZE", 3000),
            chunk_overlap=CONF.get("CHUNK_OVERLAP", 300)
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONF.get("CHUNK_SIZE", 3000),
            chunk_overlap=CONF.get("CHUNK_OVERLAP", 300),
            separators=["\n\n", "\n", " ", ""]
        )

    def _ensure_knowledge_dir(self):
        if not os.path.exists(self.default_folder):
            os.makedirs(self.default_folder)
            logger.info(f"Initialized storage: {self.default_folder}")

    def _get_file_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def _enrich_metadata(self, text_sample: str) -> dict:
        """Uses native ollama.chat() with strict JSON output."""
        prompt = f"""
        Analyze this document snippet and return a JSON object.
        CATEGORIES: [Textbook, Deep Textbook, Research Paper, Journals, Resume, SOP, Technical Report, General]
        
        Text: {text_sample[:6000]}
        
        Return ONLY a JSON object:
        {{
        "category": "Choose one from categories above",
        "keywords": ["first 5 words that desribe the genre of the doc, eg: chemistry, physics, Law, mythology, story, data_log. followed by 20 specific technical importance words"]
        }}
        """

        messages=[{'role': 'user', 'content': prompt}]
           
        try:
        # 1. Use the client to get a response
            client = ollama.AsyncClient()
            response = await client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json', # Forces the model to output valid JSON
                options={'temperature': 0},
                think=False,
            )
            
            # 2. Parse the content
            content = response['message']['content']

            # 2. Parse the content
            data = json.loads(content)
            
            return {
                "category": data.get("category", "General"),
                "keywords": ", ".join(data.get("keywords", []))
            }
        except Exception as e:
            # CRITICAL: This will tell you exactly why you're getting 'General'
            logger.error(f"Metadata Enrichment Error: {e}")
            return {"category": "General", "keywords": "error_fallback"}
        
    async def _process_file(self, file_path: str):
        file_hash = self._get_file_hash(file_path)
        filename = os.path.basename(file_path)
        
        if self.registry.is_duplicate(file_hash):
            logger.info(f"{Colors.YELLOW}Skipping duplicate: {filename}{Colors.ENDC}")
            return []

        _, ext = os.path.splitext(filename.lower())
        chunks = []

        try:
            if ext == ".pdf":
                chunks = self.pdf_engine.extract_and_chunk(file_path, mode=self.mode)
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path)
                chunks = self.text_splitter.split_documents(loader.load())
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
                chunks = self.text_splitter.split_documents(loader.load())

            if chunks:
                # Use a sample from the first few chunks for the LLM
                sample_text = " ".join([c.page_content for c in chunks[:5]])
                enriched = await self._enrich_metadata(sample_text)

                doc_metadata = {
                    "doc_id": hashlib.md5(filename.encode()).hexdigest()[:10],
                    "doc_name": filename,
                    "doc_path": file_path,
                    "doc_type": ext,
                    "doc_category": enriched["category"],
                    "keywords": enriched["keywords"],
                    "file_hash": file_hash,
                    "date_ingested": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "chunk_count": len(chunks)
                }
                
                self.registry.register_doc(doc_metadata)
                
                # Tag chunks with new metadata fields
                for chunk in chunks:
                    chunk.metadata.update({
                        "doc_id": doc_metadata["doc_id"],
                        "category": doc_metadata["doc_category"],
                        "keywords": doc_metadata["keywords"]
                    })

            return chunks
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return []

    async def load_source(self, path: str = None) -> List:
        target_path = os.path.abspath(path) if path else self.default_folder
        all_chunks = []

        if os.path.isfile(target_path):
            return await self._process_file(target_path)
        elif os.path.isdir(target_path):
            for root, _, files in os.walk(target_path):
                for file in files:
                    if file.lower().endswith((".pdf", ".docx", ".doc", ".txt")):
                        res = await self._process_file(os.path.join(root, file))
                        all_chunks.extend(res)
        return all_chunks

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to ingest")
    args = parser.parse_args()

    processor = DocumentProcessor()
    source = args.path or processor.default_folder
    
    logger.info(f"{Colors.BLUE}Starting Ingestion with Ollama Library: {source}{Colors.ENDC}")
    documents = await processor.load_source(source)
    
    if not documents:
        logger.info("No new documents to index.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name=CONF.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"),
        model_kwargs={'device': CONF.get("DEVICE", "cpu")}
    )

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CONF.get("CHROMA_PERSIST_DIR", "./knowledge_base"),
        collection_name=CONF.get("COLLECTION_NAME", "knowledge_base")
    )
    logger.info(f"{Colors.GREEN}Successfully indexed {len(documents)} chunks.{Colors.ENDC}")

if __name__ == "__main__":
    asyncio.run(main())