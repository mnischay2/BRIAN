import json
import os
import re
import asyncio
from typing import Dict, Any, Optional, Union, List

try:
    from scripts.configs.config import CONF
    from scripts.rag_retriever import RAG_Retriever
    from scripts.searcher import SearchEngine
    from scripts.scraper import WebScraper
    from scripts.pdf_reader import AdvancedPDFProcessor
    from scripts.logger import setup_aether_logger, Colors
except ImportError:
    from configs.config import CONF
    from rag_retriever import RAG_Retriever
    from searcher import SearchEngine
    from scraper import WebScraper
    from pdf_reader import AdvancedPDFProcessor
    from logger import setup_aether_logger, Colors

# ── guard: ensure Colors attributes exist even if logger stub is minimal ───────
for _attr in ("BLUE", "GREEN", "YELLOW", "RED", "BOLD", "ENDC"):
    if not hasattr(Colors, _attr):
        setattr(Colors, _attr, "")

logger = setup_aether_logger("AetherAgent")

_TOOL_TIMEOUT: int  = CONF.get("TIMEOUT_FETCH", 10)
_MAX_TOOL_CHARS: int = CONF.get("MAX_CONTENT_CHARS", 15000)
_MAX_PDF_CHUNKS: int = CONF.get("MAX_PDF_CHUNKS", 30)


class _Lazy:
    """Instantiates its object on first .get() call; logs failures gracefully."""

    def __init__(self, name: str, factory):
        self._name    = name
        self._factory = factory
        self._obj     = None
        self._failed  = False

    def get(self):
        if self._failed:
            return None
        if self._obj is None:
            try:
                self._obj = self._factory()
                logger.info(f"{Colors.GREEN}[init]{Colors.ENDC} {self._name} ready.")
            except Exception as exc:
                logger.error(
                    f"{Colors.RED}[init]{Colors.ENDC} "
                    f"{self._name} failed: {exc}"
                )
                self._failed = True
        return self._obj


_lazy_rag     = _Lazy("RAG_Retriever", RAG_Retriever)
_lazy_searcher = _Lazy("SearchEngine",  SearchEngine)
_lazy_scraper = _Lazy("WebScraper",    WebScraper)
_lazy_pdf     = _Lazy(
    "AdvancedPDFProcessor",
    lambda: AdvancedPDFProcessor(
        chunk_size=CONF.get("CHUNK_SIZE", 2000),
        chunk_overlap=CONF.get("CHUNK_OVERLAP", 300),
    ),
)


class Tools:
    """Container for all agent tools, their registry, and schema."""

    # ── tool-call parsing patterns (class-level) ───────────────────────────────
    _TC_PATTERNS = [
        re.compile(r"```tool_call\s*(\{.*?\})\s*```",        re.DOTALL | re.IGNORECASE),
        re.compile(r"```\s*(\{\s*\"tool\"\s*:.*?\})\s*```",  re.DOTALL),
        re.compile(r"(\{\s*\"tool\"\s*:.*?\"args\"\s*:.*?\})", re.DOTALL),
    ]

    def __init__(self):
        # Build the registry first so _tools_schema_str() can read it.
        self.registry: Dict[str, Dict[str, Any]] = {
            "rag_search": {
                "fn": self.tool_rag_search,
                "description": (
                    "Search the local knowledge-base (vector store). "
                    "Automatically runs multiple refined sub-queries for better recall. "
                    "Use for ingested documents and private knowledge; for docs returned by list_rag_docs. "
                    "sample: rag_search(query=['skills', 'experience'], top_k=5, doc_category='Resume', phases=2) "
                    "sample: rag_search(query='nomenclature of haloalkanes', top_k=10, doc_category='Textbook') "
                    "NEVER guess doc_name — call list_rag_docs first to get the real filename. "
                    "CATEGORIES: [Textbook, Deep Textbook, Research Paper, Journals, Resume, SOP, Technical Report, General]"
                ),
                "parameters": {
                    "query":        {"type": "string | list[string]", "required": True,  "description": "Search query or list of sub-queries"},
                    "top_k":        {"type": "integer",               "required": False, "description": "Max results per sub-query"},
                    "phases":       {"type": "integer",               "required": False, "description": "Number of sub-queries (default 2)"},
                    "doc_name":     {"type": "string",                "required": False, "description": "Filter by exact document name (use list_rag_docs first)"},
                    "doc_category": {"type": "string",                "required": False, "description": "Filter by category"},
                },
            },
            "web_search": {
                "fn": self.tool_web_search,
                "description": (
                    "Search the public internet via DuckDuckGo. "
                    "Runs multiple refined sub-queries automatically. "
                    "Use for current events, external facts, or topics not in the knowledge-base. "
                    "Each sub-query MUST be a complete meaningful phrase — no bare filter tokens like 'remote' or 'date:2026'."
                ),
                "parameters": {
                    "query":       {"type": "string | list[string]", "required": True,  "description": "Search query or list of sub-queries"},
                    "max_results": {"type": "integer",               "required": False, "description": "Max results per sub-query"},
                    "phases":      {"type": "integer",               "required": False, "description": "Number of sub-queries (default 2)"},
                },
            },
            "scrape_url": {
                "fn": self.tool_scrape_url,
                "description": (
                    "Scrape full text from one or more URLs in a single call. "
                    "Pass all URLs as a list — each call costs one tool budget. "
                    "sample: scrape_url(url=['https://a.com', 'https://b.com']) "
                    "sample: scrape_url(url='https://single.com')"
                ),
                "parameters": {
                    "url": {"type": "string | list[string]", "required": True, "description": "One URL or a list of URLs to scrape"},
                },
            },
            "read_pdf": {
                "fn": self.tool_read_pdf,
                "description": (
                    "Extract and read text from a PDF (local path or URL). "
                    "Use when given a PDF path/URL or when a search result links to a PDF. "
                    "sample: read_pdf(source='path/to/document.pdf', mode='simple')"
                ),
                "parameters": {
                    "source": {"type": "string", "required": True,  "description": "Local file path or direct URL to the PDF"},
                    "mode":   {"type": "string", "required": False, "description": "'advanced' (OCR+tables) or 'simple'"},
                },
            },
            "list_rag_docs": {
                "fn": self.tool_list_rag_docs,
                "description": (
                    "List all documents currently indexed in the local knowledge-base. "
                    "Call this BEFORE using rag_search with a doc_name filter. "
                    "sample: list_rag_docs()"
                ),
                "parameters": {},
            },
        }
        self.schema_str: str = self._tools_schema_str()

    # ── schema & parsing helpers ───────────────────────────────────────────────

    def _tools_schema_str(self) -> str:
        """Build a human-readable schema string for the system prompt."""
        lines = []
        for name, meta in self.registry.items():
            params = ", ".join(
                f"{p}({'required' if v['required'] else 'optional'}): {v['description']}"
                for p, v in meta["parameters"].items()
            )
            lines.append(f"• {name}({params})\n  → {meta['description']}")
        return "\n".join(lines)

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a tool call dict from an LLM response string."""
        for pat in self._TC_PATTERNS:
            m = pat.search(text)
            if m:
                try:
                    data = json.loads(m.group(1))
                    if isinstance(data, dict) and "tool" in data and "args" in data:
                        return data
                except json.JSONDecodeError:
                    continue
        return None

    # ── tool implementations ───────────────────────────────────────────────────

    async def tool_rag_search(
        self,
        query:        Union[str, List[str]],
        top_k:        Optional[int] = None,
        doc_name:     Optional[str] = None,
        doc_category: Optional[str] = None,
        phases:       Optional[int] = None,
    ) -> str:
        """Multi-phase RAG search over the local vector store."""
        rag = _lazy_rag.get()
        if rag is None:
            return json.dumps({"status": "error", "detail": "RAG_Retriever unavailable."})

        k         = top_k or CONF.get("RAG_TOP_K", 6)
        threshold = CONF.get("RAG_SCORE_THRESHOLD", 0.7)
        loop      = asyncio.get_running_loop()

        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, list):
            queries = query
        else:
            queries = [str(query)]

        seen:   set  = set()
        merged: list = []

        for sq in queries:
            try:
                results = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda q=sq: rag.search_knowledge_base(
                            query=q, top_k=k, s_threshold=threshold,
                            doc_name=doc_name, doc_category=doc_category,
                        ),
                    ),
                    timeout=_TOOL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[rag_search] Timeout on sub-query: {sq!r}")
                continue
            except Exception as exc:
                logger.error(f"[rag_search] Error on {sq!r}: {exc}")
                continue
            for r in results:
                sig = r["content"][:120]
                if sig not in seen:
                    seen.add(sig)
                    merged.append(r)

        if not merged:
            return json.dumps({"status": "no_results", "query": query})

        slim = [
            {
                "source":     r["source"],
                "page":       r["page"],
                "category":   r["category"],
                "similarity": r["similarity"],
                "content":    r["content"][:_MAX_TOOL_CHARS],
            }
            for r in merged
        ]
        logger.info(f"[rag_search] {len(slim)} unique chunks | sub_queries={query}")
        return json.dumps({"status": "ok", "sub_queries": query, "results": slim})

    async def tool_web_search(
        self,
        query:       Union[str, List[str]],
        phases:      Optional[int] = None,
        max_results: Optional[int] = None,
    ) -> str:
        """Multi-phase web search via DuckDuckGo."""
        searcher = _lazy_searcher.get()
        if searcher is None:
            return json.dumps({"status": "error", "detail": "SearchEngine unavailable."})

        n    = max_results or CONF.get("WEB_RESULTS_per_query", 10)
        loop = asyncio.get_running_loop()

        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, list):
            queries = query
        else:
            queries = [str(query)]

        seen:   set  = set()
        merged: list = []

        for sq in queries:
            try:
                results = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda q=sq: searcher.search(q, max_results=n)),
                    timeout=_TOOL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[web_search] Timeout on sub-query: {sq!r}")
                continue
            except Exception as exc:
                logger.error(f"[web_search] Error on {sq!r}: {exc}")
                continue
            for r in results:
                href = r.get("href", "")
                if href not in seen:
                    seen.add(href)
                    merged.append(r)

        if not merged:
            return json.dumps({"status": "no_results", "query": query})

        logger.info(f"[web_search] {len(merged)} unique results | sub_queries={query}")
        return json.dumps({"status": "ok", "sub_queries": query, "results": merged})

    async def tool_scrape_url(self, url: Union[str, List[str]]) -> str:
        """
        Scrape one or more URLs concurrently.
        Multiple URLs are scraped in a single call (counts as one tool budget).
        """
        scraper = _lazy_scraper.get()
        if scraper is None:
            return json.dumps({"status": "error", "detail": "WebScraper unavailable."})

        urls: List[str] = [url] if isinstance(url, str) else list(url)

        async def _fetch_one(u: str):
            try:
                content = await asyncio.wait_for(scraper.scrape(u), timeout=30)
            except asyncio.TimeoutError:
                return u, {"status": "timeout"}
            except Exception as exc:
                return u, {"status": "error", "detail": str(exc)}
            if not content:
                return u, {"status": "failed"}
            trimmed = content[:_MAX_TOOL_CHARS]
            logger.info(f"[scrape_url] {len(trimmed)} chars ← {u}")
            return u, {"status": "ok", "content": trimmed}

        pairs   = await asyncio.gather(*[_fetch_one(u) for u in urls])
        results = dict(pairs)
        ok_count = sum(1 for v in results.values() if v["status"] == "ok")
        logger.info(f"[scrape_url] {ok_count}/{len(urls)} URLs scraped successfully")
        return json.dumps({"status": "ok", "results": results})

    async def tool_read_pdf(self, source: str, mode: str = "simple") -> str:
        """Extract and chunk text from a PDF (local path or URL)."""
        pdf_proc = _lazy_pdf.get()
        if pdf_proc is None:
            return json.dumps({"status": "error", "detail": "PDFProcessor unavailable."})
        try:
            loop = asyncio.get_running_loop()
            docs = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: pdf_proc.extract_and_chunk(source, mode=mode)),
                timeout=120,
            )
        except asyncio.TimeoutError:
            return json.dumps({"status": "timeout", "source": source})
        except Exception as exc:
            return json.dumps({"status": "error", "source": source, "detail": str(exc)})

        if not docs:
            return json.dumps({"status": "no_content", "source": source})

        total = len(docs)
        shown = docs[:_MAX_PDF_CHUNKS]
        fname = os.path.basename(source)
        chunks = [
            {
                "source":  fname,
                "page":    d.metadata.get("page_number", d.metadata.get("page", "?")),
                "content": d.page_content[:_MAX_TOOL_CHARS],
            }
            for d in shown
        ]
        logger.info(f"[read_pdf] {len(chunks)}/{total} chunks from: {source}")
        return json.dumps({
            "status":       "ok",
            "source":       source,
            "total_chunks": total,
            "shown_chunks": len(chunks),
            "note": f"Showing first {len(chunks)} of {total} chunks." if total > _MAX_PDF_CHUNKS else "",
            "chunks":       chunks,
        })

    async def tool_list_rag_docs(self) -> str:
        """List all documents indexed in the vector store."""
        rag = _lazy_rag.get()
        if rag is None:
            return json.dumps({"status": "error", "detail": "RAG_Retriever unavailable."})
        try:
            loop = asyncio.get_running_loop()
            docs = await asyncio.wait_for(
                loop.run_in_executor(None, rag.get_docs_in_db),
                timeout=_TOOL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return json.dumps({"status": "timeout"})
        except Exception as exc:
            logger.error(f"[list_rag_docs] Unexpected error: {exc}")
            return json.dumps({"status": "error", "detail": str(exc)})
        if not docs:
            return json.dumps({"status": "empty", "docs": []})
        logger.info(f"[list_rag_docs] {len(docs)} docs in vector store")
        return json.dumps({"status": "ok", "docs": docs})
