import os
import csv
import sqlite3
from typing import List, Dict, Any

# LangChain & Vector DB Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Internal Imports
try:
    from scripts.configs.config import CONF
    from scripts.logger import setup_aether_logger, Colors
    from scripts.silence_chatter import silence
    
except ImportError:
    from configs.config import CONF
    from logger import setup_aether_logger, Colors
    from silence_chatter import silence

silence()  # Mute noisy libraries globally before anything else imports them
# Initialize Logger
logger = setup_aether_logger("DB_View")

class AetherDatabaseViewer:
    """
    Utility to extract and dump both the Chroma Vector Store (chunks) 
    and the SQLite Registry (document metadata) to CSV.
    """
    def __init__(self):
        # Config for Chroma
        self.persist_directory = CONF.get("CHROMA_PERSIST_DIR", "./knowledge_base")
        self.collection_name = CONF.get("COLLECTION_NAME", "knowledge_base")
        self.embedding_model_name = CONF.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
        
        # Config for SQLite
        self.registry_db = "aether_registry.db"
        
        # Initialize Embeddings
        logger.info(f"Loading embeddings for DB access...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': CONF.get("DEVICE", "cpu")}
        )
        
        self.vector_store = self._connect_to_chroma()

    def _connect_to_chroma(self):
        """Connects to the local Chroma instance."""
        if not os.path.exists(self.persist_directory):
            logger.error(f"Chroma directory {self.persist_directory} not found.")
            return None
        try:
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            logger.error(f"Failed to connect to Chroma: {e}")
            return None

    def export_registry_to_csv(self, output_file: str = "dumps/document_registry.csv"):
        """Extracts the doc_registry table from SQLite."""
        if not os.path.exists(self.registry_db):
            logger.warning("SQLite registry database not found. Skipping.")
            return

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            with sqlite3.connect(self.registry_db) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM doc_registry")
                rows = cursor.fetchall()

                if not rows:
                    logger.warning("Registry table is empty.")
                    return

                headers = rows[0].keys()
                with open(output_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows([dict(row) for row in rows])
                
                logger.info(f"{Colors.GREEN}Successfully exported Registry to {output_file}{Colors.ENDC}")
        except Exception as e:
            logger.error(f"Error exporting Registry: {e}")

    def export_vector_store_to_csv(self, output_file: str = "dumps/vector_chunks.csv"):
        """Extracts all chunks and metadata from Chroma."""
        if not self.vector_store:
            logger.error("Vector store not connected.")
            return

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            # Fetch all data from Chroma
            data = self.vector_store.get()
            ids = data.get('ids', [])
            docs = data.get('documents', [])
            metadatas = data.get('metadatas', [])

            if not docs:
                logger.warning("No chunks found in Chroma.")
                return

            # Determine headers from unique metadata keys
            meta_keys = set()
            for m in metadatas: meta_keys.update(m.keys())
            headers = ["id", "content"] + sorted(list(meta_keys))

            with open(output_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for i in range(len(docs)):
                    row = {"id": ids[i], "content": docs[i]}
                    row.update(metadatas[i])
                    writer.writerow(row)

            logger.info(f"{Colors.GREEN}Successfully exported Chroma to {output_file}{Colors.ENDC}")
        except Exception as e:
            logger.error(f"Error exporting Chroma: {e}")

    def run_full_export(self):
        """Runs both exports sequentially."""
        print(f"\n{Colors.BLUE}--- Aether DB Export Started ---{Colors.ENDC}")
        self.export_registry_to_csv()
        self.export_vector_store_to_csv()
        print(f"{Colors.BLUE}--- Export Complete ---{Colors.ENDC}\n")

if __name__ == "__main__":
    viewer = AetherDatabaseViewer()
    viewer.run_full_export()