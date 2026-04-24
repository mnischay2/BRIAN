import os
from typing import List, Dict, Any, Optional

# 1. FORCE OFFLINE MODE — must be set before importing transformers/langchain
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["ANONYMIZED_TELEMETRY"]  = "False"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from silence_chatter import silence
except ImportError:
    from scripts.silence_chatter import silence

silence()  

try:
    from configs.config import CONF
except ImportError:
    from scripts.configs.config import CONF

try:
    from logger import setup_aether_logger
except ImportError:
    from .logger import setup_aether_logger


logger = setup_aether_logger("RAG_Retriever")


class RAG_Retriever:

    def __init__(self):
        self.persist_directory    = CONF.get("CHROMA_PERSIST_DIR",    "./knowledge_base")
        self.collection_name      = CONF.get("COLLECTION_NAME",       "knowledge_base")
        self.embedding_model_name = CONF.get("EMBEDDING_MODEL_NAME",  "sentence-transformers/all-mpnet-base-v2")
        self.device               = CONF.get("DEVICE",                "cpu")
        self.default_top_k        = CONF.get("RAG_TOP_K",              5)

        self.similarity_threshold = CONF.get("RAG_SCORE_THRESHOLD", 0.75)

        logger.info(
            f"Loading local embedding model: "
            f"{self.embedding_model_name} on {self.device}"
        )
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={
                    "device":           self.device,
                    "local_files_only": True,
                },
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            logger.error(
                f"Failed to load local embeddings. "
                f"Ensure the model is downloaded. Error: {e}"
            )
            raise

        self.vector_store: Optional[Chroma] = None

        if os.path.exists(self.persist_directory):
            self._initialize_vector_store()
        else:
            logger.warning(
                f"Vector store directory '{self.persist_directory}' not found. "
                "Run ingestion before querying."
            )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _initialize_vector_store(self) -> None:
        try:
            store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

            count = store._collection.count()
            if count == 0:
                logger.warning(
                    f"Vector store '{self.collection_name}' is empty (0 documents). "
                    "Ingest documents before querying."
                )
                self.vector_store = store
                return

            self.vector_store = store
            logger.info(
                f"Connected to vector store '{self.collection_name}' "
                f"({count} documents)."
            )

        except Exception as e:
            logger.error(f"Vector store connection failed: {e}")
            self.vector_store = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_docs_in_db(self) -> List[Dict[str, Any]]:
        """
        Retrieves a unique list of all document names and their metadata categories
        currently indexed in the vector store.
        """
        if not self.vector_store:
            return []
        
        try:
            data = self.vector_store.get()
            metadatas = data.get('metadatas', [])
            
            seen_docs = {}
            for meta in metadatas:
                # Using doc_name from your new enrichment strategy
                name = meta.get("doc_name") or meta.get("filename", "Unknown")
                if name not in seen_docs:
                    seen_docs[name] = {
                        "doc_name": name,
                        "category": meta.get("category", "General"),
                        "doc_id":   meta.get("doc_id", "N/A")
                    }
            
            return list(seen_docs.values())
        except Exception as e:
            logger.error(f"Error fetching document list: {e}")
            return []

    def search_knowledge_base(
        self,
        query:        str,
        top_k:        Optional[int]   = None,
        s_threshold:  Optional[float] = None,
        doc_name:     Optional[str]   = None,    # New filter argument
        doc_category: Optional[str]   = None     # New filter argument
    ) -> List[Dict[str, Any]]:

        if not self.vector_store:
            self._initialize_vector_store()
            if not self.vector_store:
                logger.error("Vector store unavailable.")
                return []

        k = top_k if top_k is not None else self.default_top_k
        if s_threshold is None:
            s_threshold = self.similarity_threshold

        # --- Build Metadata Filter ---
        where_filter = {}
        filters = []
        
        if doc_name:
            filters.append({"filename": doc_name})
        if doc_category:
            filters.append({"category": doc_category})
            
        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        logger.info(
            f"Searching '{self.collection_name}' | "
            f"query={query!r} | top_k={k} | threshold={s_threshold} | filter={where_filter}"
        )

        try:
            # similarity_search_with_score supports the 'filter' kwarg (mapped to 'where' in Chroma)
            results_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=k,
                filter=where_filter if where_filter else None
            )

            formatted: List[Dict[str, Any]] = []

            for doc, raw_score in results_with_scores:
                raw_score = float(raw_score)
                # Standard Chroma normalization (depends on distance metric, usually L2)
                norm_score = (raw_score + 1) / 2
                    
                if raw_score < 0:
                    continue

                if norm_score < s_threshold:
                    continue

                formatted.append({
                    "content":    doc.page_content,
                    "metadata":   doc.metadata,
                    "source":     doc.metadata.get("doc_name") or doc.metadata.get("filename", "Unknown"),
                    "page":       doc.metadata.get("page_number") or doc.metadata.get("page", "N/A"),
                    "category":   doc.metadata.get("category", "General"),
                    "similarity": round(norm_score, 4),
                    "raw_score":  round(raw_score, 4),
                })

            logger.info(
                f"Retrieval complete: {len(formatted)}/{len(results_with_scores)} chunks passed."
            )
            return formatted

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
        
    def get_collection_info(self) -> Dict[str, Any]:
        if not self.vector_store:
            return {"status": "not_connected"}
        try:
            count = self.vector_store._collection.count()
            return {
                "status":           "connected",
                "collection":       self.collection_name,
                "document_count":   count,
                "persist_dir":      self.persist_directory,
                "embedding_model":  self.embedding_model_name,
                "threshold":        self.similarity_threshold,
                }
        except Exception as e:
            return {"status": "error", "detail": str(e)}


if __name__ == "__main__":
    rag = RAG_Retriever()
    info = rag.get_collection_info()
    print("\n=== Collection Info ===")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test doc listing
    docs = rag.get_docs_in_db()
    if docs:
        print("\n=== Documents in DB ===")
        for d in docs:
            print(f"  • {d['doc_name']} [{d['category']}]")

    if info.get("document_count", 0) == 0:
        print("\n⚠  Collection is empty — ingest documents before querying.")
    else:
        user_query = input("\nEnter your query: ").strip()
        user_filter_doc = input("Filter by document name (optional): ").strip() or None
        user_filter_cat = input("Filter by category (optional): ").strip() or None
        if user_query:
            hits = rag.search_knowledge_base(user_query, doc_name= user_filter_doc, doc_category=user_filter_cat) if user_filter_doc or user_filter_cat else None
            if not hits:
                print(f"\nNo results found.")
            else:
                print(f"\n{len(hits)} result(s) found:\n")
                for i, hit in enumerate(hits):
                    print(f"[{i+1}] {hit['source']} ({hit['category']}) sim={hit['similarity']:.4f}   Content(): {hit['content']}")
