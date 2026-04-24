"""
aether_agent.py — AetherAgent v2
=================================
Architecture: Pure ReAct loop (Reason → Act → Observe → Reason → … → Answer)

Design Principles
-----------------
• No LangGraph — eliminates state graph complexity and side-effect pitfalls
• Immutable message history — append-only each turn, never modify in-place
• Single source of truth — all state in a single RunState dataclass
• Tool calls parsed with XML tags (more reliable than regex + JSON)
• Budget management checked at loop entry, not scattered everywhere
• Session passed as parameter, not mounted on function objects
• Single responsibility per layer: think / act / observe are independent

ReAct Loop
----------
    ┌─────────────────────────────────────┐
    │  1. think(messages) → LLM response  │
    │     ├─ has <tool_call>  → act()     │
    │     └─ no <tool_call>   → final answer │
    │  2. act(tool_call)  → tool result   │
    │  3. observe: append to message history │
    │  4. go to step 1                    │
    └─────────────────────────────────────┘

Tool List
---------
  rag_search     — local vector DB retrieval (multi-sub-query)
  web_search     — DuckDuckGo search (multi-sub-query)
  scrape_url     — web page scraping (batch)
  read_pdf       — PDF extraction (local path or URL)
  list_rag_docs  — list indexed documents
  analyse        — LLM structured analysis
  synthesise     — LLM multi-source synthesis
"""

__all__ = ["AetherAgent"]

# ── Standard library ──────────────────────────────────────────────────────────
import asyncio
import dataclasses
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ── Project logging ───────────────────────────────────────────────────────────
try:
    from scripts.logger import setup_aether_logger, Colors
except ImportError:
    from .scripts.logger import setup_aether_logger, Colors

for _attr in ("BLUE", "GREEN", "YELLOW", "RED", "BOLD", "ENDC"):
    if not hasattr(Colors, _attr):
        setattr(Colors, _attr, "")

logger = setup_aether_logger("AetherAgent")

try:
    from scripts.silence_chatter import silence
except ImportError:
    try:
        from .scripts.silence_chatter import silence
    except ImportError:
        def silence(): pass
silence()

# ── Project dependencies ──────────────────────────────────────────────────────
try:
    from scripts.configs.config import CONF
    from scripts.rag_retriever import RAG_Retriever
    from scripts.searcher import SearchEngine
    from scripts.scraper import WebScraper
    from scripts.chat_llm import async_chat_snt_off
    from scripts.utils import refined_prompt_generator
except ImportError:
    from .scripts.configs.config import CONF
    from .scripts.rag_retriever import RAG_Retriever
    from .scripts.searcher import SearchEngine
    from .scripts.scraper import WebScraper
    from .scripts.chat_llm import async_chat_snt_off
    from .scripts.utils import refined_prompt_generator

try:
    from .scripts.pdf_reader import AdvancedPDFProcessor
except ImportError:
    from scripts.pdf_reader import AdvancedPDFProcessor


# ═══════════════════════════════════════════════════════════════════════════════
# Lazy Initialization
# ═══════════════════════════════════════════════════════════════════════════════

class _LazyInit:
    """Instantiate underlying object only on first .get() call, no retry after failure."""

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
                logger.info(f"{Colors.GREEN}[init]{Colors.ENDC} {self._name} ready")
            except Exception as exc:
                logger.error(f"{Colors.RED}[init]{Colors.ENDC} {self._name} failed: {exc}")
                self._failed = True
        return self._obj


_lazy_rag     = _LazyInit("RAG_Retriever",       RAG_Retriever)
_lazy_searcher= _LazyInit("SearchEngine",         SearchEngine)
_lazy_scraper = _LazyInit("WebScraper",           WebScraper)
_lazy_pdf     = _LazyInit(
    "AdvancedPDFProcessor",
    lambda: AdvancedPDFProcessor(
        chunk_size    = CONF.get("CHUNK_SIZE",    2000),
        chunk_overlap = CONF.get("CHUNK_OVERLAP", 300),
    ),
)


# ═══════════════════════════════════════════════════════════════════════════════
# constants
# ═══════════════════════════════════════════════════════════════════════════════

_TOOL_TIMEOUT   : int = CONF.get("TIMEOUT_FETCH",       10)
_MAX_TOOL_CHARS : int = CONF.get("MAX_CONTENT_CHARS",   15_000)
_MAX_PDF_CHUNKS : int = 30
_MAX_RETRIES    : int = 3       # Max retries after single tool call failure
_SHELL_JUNK = re.compile(
    r"(source\s+\S+activate\S*|^\s*/[^\s]+\s+|\r)",
    re.MULTILINE,
)

# Tool call XML tag parsing
_TC_OPEN  = "<tool_call>"
_TC_CLOSE = "</tool_call>"

# XML result tags
_TR_OPEN  = "<tool_result>"
_TR_CLOSE = "</tool_result>"


# ═══════════════════════════════════════════════════════════════════════════════
# Message Types (simple dataclasses, no LangChain dependency)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Msg:
    """Immutable message object. role is 'system' | 'user' | 'assistant'."""
    role:    str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


def _msgs_to_dicts(messages: List[Msg]) -> List[Dict[str, str]]:
    return [m.to_dict() for m in messages]


# ═══════════════════════════════════════════════════════════════════════════════
# Run State (immutable snapshot mode)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunState:
    """
    Complete run state for a single Q&A session.

    All fields return new objects after each iteration (dataclasses.replace),
    never modify in-place, completely eliminate shared state bugs.
    """
    messages      : List[Msg]   = field(default_factory=list)
    tool_calls    : int         = 0     # Tool calls used this Q&A session
    total_calls   : int         = 0     # Global cumulative tool calls
    final_answer  : str         = ""
    done          : bool        = False
    # Consecutive same tool call counter (loop detection)
    last_tool_sig : str         = ""
    loop_count    : int         = 0

    def append(self, msg: Msg) -> "RunState":
        """Return new state object with new message appended."""
        return dataclasses.replace(self, messages=self.messages + [msg])

    def inc_calls(self) -> "RunState":
        return dataclasses.replace(
            self,
            tool_calls = self.tool_calls  + 1,
            total_calls= self.total_calls + 1,
        )

    def with_sig(self, sig: str) -> "RunState":
        loop = (self.loop_count + 1) if sig == self.last_tool_sig else 0
        return dataclasses.replace(self, last_tool_sig=sig, loop_count=loop)

    def finish(self, answer: str) -> "RunState":
        return dataclasses.replace(self, final_answer=answer, done=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Call Parsing (XML tags, more reliable than regex + JSON)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]
    raw:  str   # Raw text snippet for debugging


def _parse_tool_call(text: str) -> Optional[ToolCall]:
    """
    Extract and parse <tool_call>…</tool_call> block from LLM output as JSON.

    Supports two formats:
      Format A (recommended):
        <tool_call>
        {"tool": "web_search", "args": {"query": "…"}}
        </tool_call>

      Format B (legacy compatible):
        ```tool_call
        {"tool": "…", "args": {…}}
        ```
    """
    # Format A: XML tag
    start = text.find(_TC_OPEN)
    if start != -1:
        end = text.find(_TC_CLOSE, start)
        if end != -1:
            raw_block = text[start + len(_TC_OPEN): end].strip()
            try:
                data = json.loads(raw_block)
                if isinstance(data, dict) and "tool" in data and "args" in data:
                    return ToolCall(
                        tool=str(data["tool"]),
                        args=data.get("args", {}),
                        raw=raw_block,
                    )
            except json.JSONDecodeError:
                pass  # Continue trying format B

    # Format B: Code block
    m = re.search(r"```tool_call\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict) and "tool" in data and "args" in data:
                return ToolCall(
                    tool=str(data["tool"]),
                    args=data.get("args", {}),
                    raw=m.group(1),
                )
        except json.JSONDecodeError:
            pass

    # Format C: Bare JSON (fallback last)
    m2 = re.search(r'(\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{.*?\})', text, re.DOTALL)
    if m2:
        try:
            data = json.loads(m2.group(1))
            if isinstance(data, dict) and "tool" in data and "args" in data:
                return ToolCall(
                    tool=str(data["tool"]),
                    args=data.get("args", {}),
                    raw=m2.group(1),
                )
        except json.JSONDecodeError:
            pass

    return None


def _strip_tool_call(text: str) -> str:
    """Remove tool call blocks from LLM output, keep remaining text."""
    # Remove XML tag block
    text = re.sub(
        r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    # Remove code block
    text = re.sub(
        r"```tool_call.*?```", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    return text.strip()


def _sanitise(text: str) -> str:
    cleaned = _SHELL_JUNK.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# Tool functions (same as original, but no global state dependency)
# ═══════════════════════════════════════════════════════════════════════════════

async def tool_rag_search(
    query: Union[str, List[str]],
    top_k: Optional[int]    = None,
    doc_name: Optional[str] = None,
    doc_category: Optional[str] = None,
) -> str:
    """Local vector DB multi-sub-query retrieval, results deduplicated by content fingerprint."""
    rag = _lazy_rag.get()
    if rag is None:
        return json.dumps({"status": "error", "detail": "RAG_Retriever unavailable"})

    k         = top_k or CONF.get("RAG_TOP_K", 6)
    threshold = CONF.get("RAG_SCORE_THRESHOLD", 0.7)
    loop      = asyncio.get_running_loop()
    queries   = [query] if isinstance(query, str) else [str(q) for q in query]

    seen: set         = set()
    merged: List[dict] = []

    for sq in queries:
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda q=sq: rag.search_knowledge_base(
                        query       = q,
                        top_k       = k,
                        s_threshold = threshold,
                        doc_name    = doc_name,
                        doc_category= doc_category,
                    ),
                ),
                timeout=_TOOL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[rag_search] sub-query timeout: {sq!r}")
            continue
        except Exception as exc:
            logger.error(f"[rag_search] sub-query error {sq!r}: {exc}")
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
            "source"    : r["source"],
            "page"      : r["page"],
            "category"  : r["category"],
            "similarity": r["similarity"],
            "content"   : r["content"][:_MAX_TOOL_CHARS],
        }
        for r in merged
    ]
    logger.info(f"[rag_search] {len(slim)} unique segments | sub_queries={queries}")
    return json.dumps({"status": "ok", "sub_queries": queries, "results": slim})


async def tool_web_search(
    query: Union[str, List[str]],
    max_results: Optional[int] = None,
) -> str:
    """DuckDuckGo multi-sub-query web search, results deduplicated by URL."""
    searcher = _lazy_searcher.get()
    if searcher is None:
        return json.dumps({"status": "error", "detail": "SearchEngine unavailable"})

    n       = max_results or CONF.get("WEB_RESULTS_per_query", 10)
    loop    = asyncio.get_running_loop()
    queries = [query] if isinstance(query, str) else [str(q) for q in query]

    seen: set         = set()
    merged: List[dict] = []

    for sq in queries:
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(None, lambda q=sq: searcher.search(q, max_results=n)),
                timeout=_TOOL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[web_search] sub-query timeout: {sq!r}")
            continue
        except Exception as exc:
            logger.error(f"[web_search] sub-query error {sq!r}: {exc}")
            continue

        for r in results:
            href = r.get("href", "")
            if href not in seen:
                seen.add(href)
                merged.append(r)

    if not merged:
        return json.dumps({"status": "no_results", "query": query})

    logger.info(f"[web_search] {len(merged)} unique results | sub_queries={queries}")
    return json.dumps({"status": "ok", "sub_queries": queries, "results": merged})


async def tool_scrape_url(url: Union[str, List[str]]) -> str:
    """Concurrently scrape one or more URLs, single call consumes one budget unit."""
    scraper = _lazy_scraper.get()
    if scraper is None:
        return json.dumps({"status": "error", "detail": "WebScraper unavailable"})

    urls = [url] if isinstance(url, str) else list(url)

    async def _fetch_one(u: str) -> Tuple[str, dict]:
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
    ok_cnt  = sum(1 for v in results.values() if v["status"] == "ok")
    logger.info(f"[scrape_url] {ok_cnt}/{len(urls)} URLs scraped successfully")
    return json.dumps({"status": "ok", "results": results})


async def tool_read_pdf(source: str, mode: str = "simple") -> str:
    """Extract PDF text from local path or direct URL and chunk, limit {_MAX_PDF_CHUNKS} chunks."""
    pdf_proc = _lazy_pdf.get()
    if pdf_proc is None:
        return json.dumps({"status": "error", "detail": "PDFProcessor unavailable"})

    try:
        loop = asyncio.get_running_loop()
        docs = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: pdf_proc.extract_and_chunk(source, mode="simple")),
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
            "source" : fname,
            "page"   : d.metadata.get("page_number", d.metadata.get("page", "?")),
            "content": d.page_content[:_MAX_TOOL_CHARS],
        }
        for d in shown
    ]
    logger.info(f"[read_pdf] {len(chunks)}/{total} chunks from: {source}")
    return json.dumps({
        "status"      : "ok",
        "source"      : source,
        "total_chunks": total,
        "shown_chunks": len(chunks),
        "note"        : f"showing first {len(chunks)}/{total} chunks" if total > _MAX_PDF_CHUNKS else "",
        "chunks"      : chunks,
    })


async def tool_list_rag_docs() -> str:
    """List all indexed documents in local vector DB."""
    rag = _lazy_rag.get()
    if rag is None:
        return json.dumps({"status": "error", "detail": "RAG_Retriever unavailable"})

    try:
        loop = asyncio.get_running_loop()
        docs = await asyncio.wait_for(
            loop.run_in_executor(None, rag.get_docs_in_db),
            timeout=_TOOL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return json.dumps({"status": "timeout"})

    if not docs:
        return json.dumps({"status": "empty", "docs": []})

    logger.info(f"[list_rag_docs] vector DB {len(docs)} documents")
    return json.dumps({"status": "ok", "docs": docs})


async def tool_analyse(context: str, focus: Optional[str] = None) -> str:
    """
    Perform LLM structured analysis on any text block.
    Returns: entities, themes, facts, gaps, summary.
    """
    if not context or not context.strip():
        return json.dumps({"status": "error", "detail": "context is empty"})

    focus_line = f"\nPlease focus on: {focus}" if focus else ""
    prompt = (
        "You are a professional analyst. Please analyze the following context, return valid JSON only,"
        "no preamble or Markdown code blocks, keys as follows：\n"
        "  entities : list[str] — key named entities (person, organization, product, location)\n"
        "  themes   : list[str] — main themes\n"
        "  facts    : list[str] — concrete verifiable facts and data points\n"
        "  gaps     : list[str] — unanswered questions or missing information\n"
        "  summary  : str       — 2-3 sentence plain text summary\n"
        f"{focus_line}\n\nContext:\n{context[:_MAX_TOOL_CHARS]}"
    )
    try:
        raw   = await async_chat_snt_off([{"role": "user", "content": prompt}])
        clean = re.sub(r"```[a-z]*|```", "", raw).strip()
        parsed= json.loads(clean)
        logger.info(
            f"[analyse] entities={len(parsed.get('entities',[]))} "
            f"themes={len(parsed.get('themes',[]))} "
            f"facts={len(parsed.get('facts',[]))}"
        )
        return json.dumps({"status": "ok", "analysis": parsed})
    except json.JSONDecodeError:
        logger.warning("[analyse] JSON parse failed — returning raw text")
        return json.dumps({"status": "ok", "analysis": {"raw": raw}})
    except Exception as exc:
        logger.error(f"[analyse] error: {exc}")
        return json.dumps({"status": "error", "detail": str(exc)})


async def tool_synthesise(sources: List[str], objective: str) -> str:
    """
    Merge multiple text sources into coherent synthesis draft, aligned to objective.
    Call after collecting from multiple tools, before writing final answer.
    """
    if not sources:
        return json.dumps({"status": "error", "detail": "sources not provided"})
    if not objective or not objective.strip():
        return json.dumps({"status": "error", "detail": "objective is empty"})

    per_limit = _MAX_TOOL_CHARS // max(len(sources), 1)
    numbered  = "\n\n".join(
        f"[source {i+1}]\n{s[:per_limit]}" for i, s in enumerate(sources)
    )
    prompt = (
        "You are a professional research synthesizer. Based on the following numbered sources, generate synthesis content meeting the objective.\n\n"
        f"Objective: {objective}\n\n{numbered}\n\n"
        "return valid JSON only, no preamble or Markdown code block：\n"
        "{\n"
        '  "synthesis"  : "<coherent paragraph covering all sources>",\n'
        '  "key_points" : ["concise key points …"],\n'
        '  "confidence" : "high|medium|low",\n'
        '  "gaps"       : ["remaining unknown information or caveats"]\n'
        "}"
    )
    try:
        raw   = await async_chat_snt_off([{"role": "user", "content": prompt}])
        clean = re.sub(r"```[a-z]*|```", "", raw).strip()
        parsed= json.loads(clean)
        logger.info(
            f"[synthesise] key_points={len(parsed.get('key_points',[]))} "
            f"confidence={parsed.get('confidence','?')}"
        )
        return json.dumps({"status": "ok", "result": parsed})
    except json.JSONDecodeError:
        logger.warning("[synthesise] JSON parse failed — returning raw content")
        return json.dumps({"status": "ok", "result": {"synthesis": raw}})
    except Exception as exc:
        logger.error(f"[synthesise] error: {exc}")
        return json.dumps({"status": "error", "detail": str(exc)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Registry (pure data, no global mutable cache)
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "rag_search": {
        "fn": tool_rag_search,
        "desc": (
            "Search local knowledge base (vector DB). Pass short, distinct sub-query list to maximize recall. "
            "For documents in knowledge base and private knowledge.\n"
            "Example: rag_search(query=['resume skills', 'work experience'], top_k=6, doc_category='Resume')\n"
            "Categories: Textbook | Deep Textbook | Research Paper | Journals | Resume | SOP | "
            "Technical Report | General (one per call)"
        ),
        "params": {
            "query"       : {"type": "str|list[str]", "required": True,  "desc": "one or more search queries"},
            "top_k"       : {"type": "int",           "required": False, "desc": "max results per sub-query (default 6)"},
            "doc_name"    : {"type": "str",           "required": False, "desc": "filter by exact filename"},
            "doc_category": {"type": "str",           "required": False, "desc": "filter by document category"},
        },
    },
    "web_search": {
        "fn": tool_web_search,
        "desc": (
            "Search internet via DuckDuckGo. Pass short, distinct sub-query list. "
            "Add year or 'latest' when recency is needed.\n"
            "Example: web_search(query=['ML engineer jobs 2025', 'Python AI remote positions 2025'])"
        ),
        "params": {
            "query"      : {"type": "str|list[str]", "required": True,  "desc": "one or more search queries"},
            "max_results": {"type": "int",           "required": False, "desc": "max results per sub-query (default 10)"},
        },
    },
    "scrape_url": {
        "fn": tool_scrape_url,
        "desc": (
            "Concurrently scrape one or more URLs in single call (consumes one budget unit). "
            "Must pass all needed URLs as list at once.\n"
            "Example: scrape_url(url=['https://a.com', 'https://b.com'])"
        ),
        "params": {
            "url": {"type": "str|list[str]", "required": True, "desc": "single URL string or list of multiple URLs"},
        },
    },
    "read_pdf": {
        "fn": tool_read_pdf,
        "desc": (
            "Extract and read PDF text from local path or direct URL. "
            "Use when search result links to PDF not yet in knowledge base.\n"
            "Example: read_pdf(source='https://example.com/paper.pdf', mode='simple')"
        ),
        "params": {
            "source": {"type": "str", "required": True,  "desc": "local file path or direct PDF link"},
            "mode"  : {"type": "str", "required": False, "desc": "'simple' (default) or 'advanced' (OCR+tables)"},
        },
    },
    "list_rag_docs": {
        "fn": tool_list_rag_docs,
        "desc": (
            "List all indexed documents in local knowledge base. "
            "Call first when user asks what documents available, or to confirm filenames before precise rag_search.\n"
            "Example: list_rag_docs()"
        ),
        "params": {},
    },
    "analyse": {
        "fn": tool_analyse,
        "desc": (
            "Perform LLM structured analysis on any text block. "
            "Return entities, themes, facts, gaps, and summary. "
            "Call after rag_search/web_search/scrape_url, before synthesise or writing final answer.\n"
            "Example: analyse(context='<raw text>', focus='skills and seniority level')"
        ),
        "params": {
            "context": {"type": "str", "required": True,  "desc": "raw text to analyze"},
            "focus"  : {"type": "str", "required": False, "desc": "optional focused analysis instruction"},
        },
    },
    "synthesise": {
        "fn": tool_synthesise,
        "desc": (
            "Synthesize multiple text sources into coherent draft. "
            "Call after collecting from multiple tools, before writing final answer.\n"
            "Example: synthesise(sources=['resume text', 'job description text'], "
            "objective='match candidate profile with job requirements')"
        ),
        "params": {
            "sources"  : {"type": "list[str]", "required": True, "desc": "list of source text strings (one per source)"},
            "objective": {"type": "str",       "required": True, "desc": "synthesis objective"},
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Dispatch (clean, no side effects)
# ═══════════════════════════════════════════════════════════════════════════════

async def dispatch_tool(tc: ToolCall) -> Tuple[str, bool]:
    """
    Dispatch tool call and return (result_json, success).

    Never throw exception — all errors wrapped as JSON string,
    let LLM see error and decide how to recover.
    """
    if tc.tool not in TOOL_REGISTRY:
        return json.dumps({
            "status": "error",
            "detail": f"Unknown tool '{tc.tool}', available tools: {list(TOOL_REGISTRY.keys())}",
        }), False

    try:
        logger.info(
            f"{Colors.YELLOW}[tool dispatch]{Colors.ENDC} "
            f"'{tc.tool}' | args={tc.args}"
        )
        result = await TOOL_REGISTRY[tc.tool]["fn"](**tc.args)
        logger.info(
            f"{Colors.GREEN}[tool complete]{Colors.ENDC} "
            f"'{tc.tool}' | {len(result)} chars"
        )
        return result, True

    except TypeError as exc:
        msg = (
            f"Tool '{tc.tool}' parameter error: {exc}. "
            "Please check parameter names and types and retry."
        )
        logger.warning(f"[tool dispatch] TypeError: {exc}")
        return json.dumps({"status": "error", "detail": msg}), False

    except Exception as exc:
        logger.error(f"[tool dispatch] '{tc.tool}' exception: {exc}", exc_info=True)
        return json.dumps({"status": "error", "detail": str(exc), "tool": tc.tool}), False


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt (generated on demand, no global cache)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_tools_doc() -> str:
    """Generate tool documentation string."""
    lines = []
    for name, meta in TOOL_REGISTRY.items():
        params = "\n    ".join(
            f"{p}({'required' if v['required'] else 'optional'}): {v['desc']}"
            for p, v in meta["params"].items()
        ) or "(no parameters)"
        lines.append(f"• **{name}**\n    {meta['desc']}\n  Parameters:\n    {params}")
    return "\n\n".join(lines)


def _build_system_prompt() -> str:
    budget      = CONF.get("DR_Tools_budgets",           20)
    max_per_q   = CONF.get("max_tool_calls_per_question", 20)
    rag_top_k   = CONF.get("RAG_TOP_K",                   6)
    threshold   = CONF.get("RAG_SCORE_THRESHOLD",         0.7)
    rag_phases  = CONF.get("RAG_PHASES",                  2)
    web_phases  = CONF.get("WEB_PHASES",                  2)
    web_results = CONF.get("WEB_RESULTS_per_query",       10)
    tools_doc   = _build_tools_doc()
    now         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""You are Aether, a smart research and reasoning agent.
Think step-by-step. Use tools strategically to answer questions accurately.

Current date/time: {now}

═══════════════════ OPERATIONAL PARAMETERS ═══════════════════
• Global tool budget              : {budget}
• Max tool calls / question       : {max_per_q}
• RAG retrieval phases            : {rag_phases} (rag_search will run the number of sub-queries you give it)
• Web search phases               : {web_phases} (web_search will run the number of sub-queries you give it)
• RAG top-k per sub-query         : {rag_top_k}
• RAG similarity threshold        : {threshold}
• Web results per sub-query       : {web_results}

═══════════════════ AVAILABLE TOOLS ═══════════════════
{tools_doc}

═══════════════════ DECISION RULES ═══════════════════
0. (MOST IMPORTANT) First analyze what the query is about. Identify potential information sources (knowledge-base vs web). Decide which tool(s) to use based on that analysis.
   - Question about specific document → prefer rag_search with doc filters
   - Question about current events/jobs/news → MUST use web_search
   - Found promising URL → use scrape_url to get content
   - Encounter PDF → use read_pdf to extract text

1. Decide whether a tool is needed before answering. Most questions require at least one tool call.

2. For knowledge-base questions:
   - First call list_rag_docs if unsure what documents exist
   - Then call rag_search with {rag_phases} distinct sub-queries covering different angles
   - rag_search handles multi-phase internally

3. For current/external information:
   - Call web_search with {web_phases} distinct sub-queries
   - web_search handles multi-phase internally

4. Use scrape_url to read the full content of promising URLs from web_search results.

5. Use read_pdf when given a PDF path/URL or when search result links to a non-ingested PDF.

6. NEVER exceed {max_per_q} tool calls for a single question.

7. Once you have enough context, STOP calling tools and write the final answer.

8. For rag_search and web_search: give EXACTLY {rag_phases} or {web_phases} queries respectively, as short distinct phrases covering different angles.

═══════════════════ TOOL CALL FORMAT ═══════════════════
When calling a tool, output ONLY this fenced block — nothing else:

```tool_call
{{
"tool": "<tool_name>",
"args": {{
    "<param1>": "<value1>"
}}
}}
```

When giving the final answer, write in plain text/markdown — NO tool_call block.

═══════════════════ IMPORTANT ═══════════════════
• Cite all sources (document name / URL) in the final answer
• If nothing is found in knowledge-base or web, state that honestly
• Keep reasoning between tool calls brief
• Do not fabricate or guess information you don't have
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ReAct Core Loop
# ═══════════════════════════════════════════════════════════════════════════════

class ReActLoop:
    """
    Pure ReAct execution engine.

    each iteration：
      1. think  — send complete message history to LLM，get response
      2. route  — check if response contains tool calls
      3a. act   — if tool calls: dispatch tool,append result as assistant + user message pair
      3b. done  — if no tool calls:set response as final answer, end loop
    """

    def __init__(self, session: "SessionManager"):
        self.session    = session
        self.sys_prompt = _build_system_prompt()
        self.budget     = CONF.get("DR_Tools_budgets",           20)
        self.max_per_q  = CONF.get("max_tool_calls_per_question", 20)

    # ── Core Reasoning Call ──────────────────────────────────────────────────────────

    async def think(self, state: RunState) -> str:
        """Call LLM, return raw response text."""
        # dynamically inject budget and loop warning to system prompt
        sys_content = self.sys_prompt
        remaining   = min(
            self.budget - state.total_calls,
            self.max_per_q - state.tool_calls,
        )

        if remaining <= 2:
            sys_content += (
                f"\n\n⚠ Budget about to be exhausted (remaining {remaining} tool calls). "
                "If sufficient information exists, immediately write final answer. Do not call any more tools.\n"
            )
        if state.loop_count >= 2:
            sys_content += (
                f"\n\n⚠ Detected loop: same tool calls repeated {state.loop_count + 1} times. "
                "Use different query, different tools, or skip this sub-task.\n"
            )

        all_msgs = [Msg("system", sys_content)] + state.messages

        logger.info(
            f"{Colors.BLUE}[think]{Colors.ENDC} "
            f"global={state.total_calls}/{self.budget} | "
            f"this question={state.tool_calls}/{self.max_per_q} | "
            f"loop={state.loop_count} | "
            f"message count={len(all_msgs)}"
        )

        response = await async_chat_snt_off(_msgs_to_dicts(all_msgs))

        if not response or not response.strip():
            logger.warning("[think] LLM returns empty response — using fallback response")
            response = "Unable to generate response, please try asking in a different way."

        return response

    # ── Tool Execution ──────────────────────────────────────────────────────────────

    async def act(self, tc: ToolCall, state: RunState) -> Tuple[RunState, str]:
        """
        Dispatch tool and append result to message history.

        Message Append Order：
          assistant: LLM thinking (contains tool call chunks)
          user:      tool result (wrapped in XML tags)

        note：assistant message was already appended by main loop before calling this function,
        here we only append user side tool result message.
        """
        # Compute tool call signature (for loop detection)
        sig   = f"{tc.tool}:{json.dumps(tc.args, sort_keys=True)}"
        state = state.with_sig(sig)

        # Log to session
        self.session.log_tool_call(
            tc.tool, tc.args,
            {"tool_calls": state.tool_calls, "total_calls": state.total_calls},
        )

        result, success = await dispatch_tool(tc)
        self.session.log_tool_result(tc.tool, result, success)

        # Build tool result message (user role, XML tags clearly annotate source)
        result_msg = Msg(
            "user",
            f"<tool_result tool=\"{tc.tool}\">\n{result}\n</tool_result>",
        )
        state = state.append(result_msg).inc_calls()

        logger.info(
            f"[act] '{tc.tool}' complete | success={success} | "
            f"budget={state.tool_calls}/{self.max_per_q}"
        )
        return state, result

    # ── Main Loop ────────────────────────────────────────────────────────────────

    async def run(self, initial_state: RunState) -> RunState:
        """
        Execute complete ReAct loop until final answer or budget exhausted.

        Return terminal RunState with final_answer.
        """
        state = initial_state

        while not state.done:
            # ── Budget check (unified at loop entry, not scattered)─────────────────────
            if state.total_calls >= self.budget:
                logger.warning(f"[loop] global budget exhausted ({state.total_calls}/{self.budget})")
                state = await self._force_answer(state, "Global tool budget exhausted")
                break
            if state.tool_calls >= self.max_per_q:
                logger.warning(f"[loop] question budget exhausted ({state.tool_calls}/{self.max_per_q})")
                state = await self._force_answer(state, "Question tool budget exhausted")
                break

            # ── Think ────────────────────────────────────────────────────────
            response = await self.think(state)

            # Append assistant message to history (whether or not contains tool call)
            state = state.append(Msg("assistant", response))
            self.session.log_llm_response(
                response,
                {"tool_calls": state.tool_calls, "total_calls": state.total_calls},
            )

            # ── Route: detect tool call ─────────────────────────────────────────
            tc = _parse_tool_call(response)

            if tc is None:
                # No tool call → final answer
                # Clean up any residual tool call tags (edge case)
                answer = _strip_tool_call(response)
                if not answer.strip():
                    # Scan history for most recent valid assistant message
                    answer = self._find_last_answer(state)

                if not answer.strip():
                    answer = "Unable to produce answer, please try asking differently."

                state = state.finish(answer)
                logger.info(
                    f"{Colors.GREEN}[loop]{Colors.ENDC} "
                    f"final answer | {len(answer)} chars | "
                    f"tool calls={state.tool_calls}"
                )
                break

            # ── Act: execute tool ────────────────────────────────────────────────
            state, _ = await self.act(tc, state)

        return state

    # ── Helper Methods ──────────────────────────────────────────────────────────────

    async def _force_answer(self, state: RunState, reason: str) -> RunState:
        """
        Force synthesize final answer when budget exhausted.
        Inject system message requiring LLM to immediately synthesize available information.
        """
        logger.info(f"[force answer] reason: {reason}")
        force_msg = Msg(
            "user",
            f"[system message] {reason}. Based on all information collected so far, "
            "immediately output complete final answer. "
            "Requirements: cite all sources (RAG: document name+page number; web: URL); "
            "use titles and bullet points; must not contain any <tool_call> chunks.",
        )
        temp_state = state.append(force_msg)
        sys_content= self.sys_prompt + "\n\n⚠ Output final answer immediately, do not call any more tools.\n"
        all_msgs   = [Msg("system", sys_content)] + temp_state.messages

        response = await async_chat_snt_off(_msgs_to_dicts(all_msgs))
        if not response or not response.strip():
            response = "Due to tool budget exhaustion, cannot complete full analysis. Please resubmit based on available information."

        answer = _strip_tool_call(response)
        return state.append(Msg("assistant", response)).finish(answer)

    @staticmethod
    def _find_last_answer(state: RunState) -> str:
        """Scan message history in reverse, find most recent assistant message without tool call."""
        for msg in reversed(state.messages):
            if msg.role == "assistant":
                if _parse_tool_call(msg.content) is None:
                    cleaned = _strip_tool_call(msg.content)
                    if cleaned.strip():
                        return cleaned
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Session Manager
# ═══════════════════════════════════════════════════════════════════════════════

class SessionManager:
    """
    Record complete event log per session, stored as
    sessions/<YYYY-MM-DD>/<session_id>.json。

    All events appended in order with timestamp and sequence number, provides complete audit trail.
    """

    def __init__(self):
        self.session_id   = self._gen_id()
        self.session_dir  = self._make_dir()
        self.session_file = self.session_dir / f"{self.session_id}.json"
        self._seq         = 0
        self._init_file()

    @staticmethod
    def _gen_id() -> str:
        return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _make_dir() -> Path:
        d = Path("sessions") / datetime.now().strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _init_file(self) -> None:
        data = {
            "session_id"          : self.session_id,
            "start_time"          : datetime.now().isoformat(),
            "model"               : CONF.get("LLM_MODEL", "unknown"),
            "budget_global"       : CONF.get("DR_Tools_budgets", 20),
            "budget_per_question" : CONF.get("max_tool_calls_per_question", 20),
            "events"              : [],
        }
        self._write(data)
        logger.info(f"[session] started: {self.session_id} → {self.session_file}")

    def _write(self, data: Dict[str, Any]) -> None:
        try:
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"[session] write failed: {exc}")

    def _append(self, event_type: str, data: Dict[str, Any]) -> None:
        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            session_data["events"].append({
                "seq"      : self._seq,
                "timestamp": datetime.now().isoformat(),
                "type"     : event_type,
                **data,
            })
            self._seq += 1
            self._write(session_data)
        except Exception as exc:
            logger.warning(f"[session] event loggingfailed: {exc}")

    # ── Public API ──────────────────────────────────────────────────────────────

    def log_user_message(self, content: str) -> None:
        self._append("user_message", {"content": content[:20_000]})

    def log_llm_response(self, content: str, budget_state: Dict[str, int]) -> None:
        self._append("llm_response", {"content": content[:20_000], "budget_state": budget_state})

    def log_tool_call(self, tool: str, args: Dict[str, Any], budget_state: Dict[str, int]) -> None:
        self._append("tool_call", {"tool": tool, "arguments": args, "budget_state": budget_state})

    def log_tool_result(self, tool: str, result: str, success: bool) -> None:
        self._append("tool_result", {"tool": tool, "success": success, "result": result[:20_000]})

    def log_question_complete(self, question: str, answer: str, tool_calls: int) -> None:
        self._append("question_complete", {
            "question"  : question[:5_000],
            "answer"    : answer[:20_000],
            "tool_calls": tool_calls,
        })

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        self._append(event_type, data)

    def finalize(self) -> None:
        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["end_time"]     = datetime.now().isoformat()
            data["total_events"] = len(data["events"])
            self._write(data)
            logger.info(f"[session] finalized: {self.session_id}")
        except Exception as exc:
            logger.warning(f"[session] finalization failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# AetherAgent Main Class
# ═══════════════════════════════════════════════════════════════════════════════

class AetherAgent:
    """
    Top-level agent class.

    Public API：
      await agent.run(user_input, use_history=False) → str
      await agent.chat()                              — Interactive REPL
      agent.reset_memory()                           — clear multi-turn history
    """

    def __init__(self):
        self.session  = SessionManager()
        self._history : List[Msg] = []   # multi-turn conversation history（only user/assistant message）

        logger.info(
            f"{Colors.GREEN}[AetherAgent]{Colors.ENDC} ready | "
            f"model={CONF.get('LLM_MODEL','unknown')} | "
            f"budget={CONF.get('DR_Tools_budgets',20)} global / "
            f"{CONF.get('max_tool_calls_per_question',20)} per question"
        )

    # ── Core Execution ──────────────────────────────────────────────────────────────

    async def run(self, user_input: str, use_history: bool = False) -> str:
        """
        End-to-end process single user query.

        Parameters
        ----
        user_input   : raw user text
        use_history  : If True, append multi-turn conversation history before current query

        Returns
        ----
        Final answer string.
        """
        clean = _sanitise(user_input)
        if not clean:
            return "Message seems empty, please re-enter."

        logger.info(f"{Colors.BLUE}[run]{Colors.ENDC} query: {clean!r}")
        self.session.log_user_message(clean)

        # Build initial message list
        history      = self._history if use_history else []
        start_msgs   = history + [Msg("user", clean)]

        initial = RunState(messages=start_msgs)

        # Instantiate ReActLoop (create new instance each run, no shared state)
        react  = ReActLoop(session=self.session)
        final  = await react.run(initial)
        answer = final.final_answer

        if not answer.strip():
            answer = "Unable to produce answer, please try asking differently."

        # Update multi-turn history (keep only user/assistant backbone, no tool results)
        self._history = self._history + [
            Msg("user",      clean),
            Msg("assistant", answer),
        ]

        self.session.log_question_complete(clean, answer, final.tool_calls)

        logger.info(
            f"{Colors.GREEN}[run]{Colors.ENDC} complete | "
            f"tool calls={final.tool_calls} | answer={len(answer)} chars"
        )
        return answer

    # ── Memory Management ──────────────────────────────────────────────────────────────

    def reset_memory(self) -> None:
        """Clear multi-turn conversation history."""
        self._history = []
        self.session.log_event("system_event", {"event": "user cleared conversation history"})
        logger.info("[AetherAgent] Conversation history cleared")

    # ── Interactive REPL ───────────────────────────────────────────────────────────

    async def chat(self) -> None:
        """
        Command-line interactive REPL.

        Commands:
          exit / quit — End session
          reset       — Clear multi-turn history
        """
        print(
            f"\n{Colors.BOLD}{Colors.BLUE}"
            "╔══════════════════════════════════╗\n"
            "║      AETHER  THINKING  AGENT     ║\n"
            "╚══════════════════════════════════╝"
            f"{Colors.ENDC}\n"
            f"  model  : {CONF.get('LLM_MODEL', 'unknown')}\n"
            f"  budget  : {CONF.get('DR_Tools_budgets', 20)} global | "
            f"{CONF.get('max_tool_calls_per_question', 20)} per question\n"
            f"  memory  : multi-turn conversation (input 'reset' to clear)\n"
            f"  session  : {self.session.session_id}\n"
            f"  Input 'exit' to exit.\n"
        )

        while True:
            try:
                raw = input(f"{Colors.BOLD}You >{Colors.ENDC} ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                self.session.finalize()
                break

            user_input = raw.strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                self.session.finalize()
                break
            if user_input.lower() == "reset":
                self.reset_memory()
                print(f"{Colors.YELLOW}[Conversation history cleared]{Colors.ENDC}\n")
                continue

            print(f"\n{Colors.YELLOW}[Thinking…]{Colors.ENDC}")
            try:
                answer = await self.run(user_input, use_history=True)
            except asyncio.CancelledError:
                print(f"\n{Colors.YELLOW}[Interrupted]{Colors.ENDC}")
                self.session.finalize()
                break
            except Exception as exc:
                logger.error(f"[chat] unexpected error: {exc}", exc_info=True)
                print(f"{Colors.RED}[error]{Colors.ENDC} {exc}\n")
                continue

            print(f"\n{Colors.GREEN}Aether >{Colors.ENDC}\n{answer}\n")
            print("─" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════════════

async def _main() -> None:
    agent = AetherAgent()
    await agent.chat()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nSession ended.")
        sys.exit(0)