import os
import sys
import logging
import tempfile
import threading
import time
import unicodedata
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import List, Literal, Tuple

os.environ["PYTHONUNBUFFERED"]    = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"]    = "4"
os.environ["HF_HUB_OFFLINE"]    = "1"

import pypdf
from contextlib import contextmanager
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document

try:
    from configs.config import CONF
except ImportError:
    from .configs.config import CONF

try:
    from logger import setup_aether_logger, Colors
except ImportError:
    from .logger import setup_aether_logger, Colors

logger = setup_aether_logger("PDFReader")

logging.getLogger("RapidOCR").setLevel(logging.ERROR)
logging.getLogger("docling").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableStructureOptions,
)


# ---------------------------------------------------------------------------
# Thread-local converter storage — one instance per thread, reused forever
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_base_converter() -> DocumentConverter:
    if not hasattr(_thread_local, "base_converter"):
        opts = PdfPipelineOptions()
        opts.do_ocr             = False
        opts.do_table_structure = True
        opts.table_structure_options = TableStructureOptions(do_cell_matching=True)
        _thread_local.base_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        logger.debug(f"[thread {threading.get_ident()}] Base converter initialised.")
    return _thread_local.base_converter


def _get_ocr_converter() -> DocumentConverter:
    if not hasattr(_thread_local, "ocr_converter"):
        opts = PdfPipelineOptions()
        opts.do_ocr             = True
        opts.ocr_options        = RapidOcrOptions()
        opts.do_table_structure = True
        opts.table_structure_options = TableStructureOptions(do_cell_matching=True)
        _thread_local.ocr_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        logger.debug(f"[thread {threading.get_ident()}] OCR converter initialised.")
    return _thread_local.ocr_converter


# ---------------------------------------------------------------------------
# Readability helpers (shared by both threshold functions)
# ---------------------------------------------------------------------------

def _readable_ratio(text: str) -> float:
    """Fraction of characters that are ASCII or standard Unicode letters."""
    if not text:
        return 0.0
    readable = sum(
        1 for c in text
        if c.isascii() or unicodedata.category(c).startswith("L")
    )
    return readable / len(text)


def _readable_word_count(text: str) -> int:
    """Words where >60% of characters are ASCII or Unicode letters."""
    count = 0
    for word in text.split():
        if not word:
            continue
        letter_chars = sum(
            c.isascii() or unicodedata.category(c).startswith("L")
            for c in word
        )
        if letter_chars / len(word) > 0.60:
            count += 1
    return count


def _avg_readable_word_length(text: str) -> float:
    """Average length of readable words."""
    lengths = []
    for word in text.split():
        if not word:
            continue
        letter_chars = sum(
            c.isascii() or unicodedata.category(c).startswith("L")
            for c in word
        )
        if len(word) > 0 and letter_chars / len(word) > 0.60:
            lengths.append(len(word))
    return sum(lengths) / max(len(lengths), 1)


# ---------------------------------------------------------------------------
# Two-tier readability checks — THE FIX
# ---------------------------------------------------------------------------

def is_encoding_garbage(text: str) -> Tuple[bool, str]:
    """
    PRE-OCR check: Is this Docling output encoding garbage that needs OCR?

    Calibrated to catch symbol soup (✬✗✘✒✎✗✙) WITHOUT falsely triggering
    on sparse-but-valid pages:
      • Title pages         (5-15 readable words)  → OCR triggered, not skipped
      • Figure caption pages (8-25 words)          → OCR triggered, not skipped
      • Formula pages       (short tokens)         → OCR triggered, not skipped
      • Table pages                                → OCR triggered, not skipped

    Thresholds:
      ratio    < 0.50  — most chars are symbols (pure garbage)
      words    < 15    — very few real words (encoding broken or actually empty)
      avg_len  < 4.5   — short split-artefacts like "Coo Coor C oo Coordina ion"
                         (real sparse pages like "NaCl HCl H2O" have avg≈3.5,
                          which triggers OCR — that is intentional and harmless:
                          OCR on a clean formula page just returns the same text)
    """
    clean = text.strip()
    if not clean:
        return True, "empty"

    ratio   = _readable_ratio(clean)
    rwords  = _readable_word_count(clean)
    avg_len = _avg_readable_word_length(clean)

    if ratio   < 0.50: return True, f"ratio={ratio:.2f} < 0.50"
    if rwords  < 15:   return True, f"words={rwords} < 15"
    if avg_len < 4.5:  return True, f"avg_len={avg_len:.2f} < 4.5"

    return False, f"ok (ratio={ratio:.2f}, words={rwords}, avg_len={avg_len:.2f})"


def is_completely_unreadable(text: str) -> Tuple[bool, str]:
    """
    POST-OCR check: Is this OCR output so bad it's completely unusable?

    Extremely permissive — preserves virtually everything OCR produces,
    including sparse pages (title pages, captions, formula pages).

    Only discards:
      • Truly empty output
      • Pure symbol garbage that OCR somehow reproduced (ratio < 0.40)
      • Fewer than 3 real words (essentially blank)
      • Space-separated single chars "T h e n a m e s" (avg_len < 2.0)
        which are unembeddable and useless for RAG

    Crucially, this does NOT discard:
      • Title pages with 5 words
      • Figure pages with 10 words
      • Any page with actual words regardless of count
    """
    clean = text.strip()
    if not clean:
        return True, "empty"

    ratio   = _readable_ratio(clean)
    rwords  = _readable_word_count(clean)
    avg_len = _avg_readable_word_length(clean)

    if ratio   < 0.40: return True, f"ratio={ratio:.2f} < 0.40"
    if rwords  < 3:    return True, f"words={rwords} < 3"
    if avg_len < 2.0:  return True, f"avg_len={avg_len:.2f} < 2.0"

    return False, f"has_content (ratio={ratio:.2f}, words={rwords}, avg_len={avg_len:.2f})"


# ---------------------------------------------------------------------------
# Stdout suppressor
# ---------------------------------------------------------------------------

@contextmanager
def suppress_internal_noise():
    old_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = old_stdout


# ---------------------------------------------------------------------------
# OCR worker — runs in persistent ocr_pool
# ---------------------------------------------------------------------------

def _ocr_single_page(
    seg_path:        str,
    local_idx:       int,   # 1-based local Docling index
    global_page_num: int,
    seg_idx:         int,
) -> str:
    """
    OCR a single page. Uses this thread's cached OCR converter.
    Returns text if readable, empty string if completely unreadable.
    """
    tmp_path = None
    try:
        tmp_path = tempfile.mktemp(suffix=".pdf")
        reader = pypdf.PdfReader(seg_path)
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[local_idx - 1])   # local_idx is 1-based
        with open(tmp_path, "wb") as f:
            writer.write(f)

        with suppress_internal_noise():
            result = _get_ocr_converter().convert(tmp_path)
        ocr_md = result.document.export_to_markdown()

        # Post-OCR: only discard completely unreadable output
        bad, reason = is_completely_unreadable(ocr_md)
        if bad:
            logger.warning(
                f"[seg {seg_idx}] p.{global_page_num}: "
                f"OCR output completely unreadable ({reason}) — page skipped."
            )
            return ""

        return ocr_md

    except Exception as e:
        logger.error(
            f"[seg {seg_idx}] p.{global_page_num}: _ocr_single_page raised: {e}"
        )
        return ""

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Segment worker — runs in persistent segment_pool
# ---------------------------------------------------------------------------

def _process_segment(
    seg_idx:       int,
    seg_start:     int,
    seg_path:      str,
    source:        str,
    num_pages:     int,
    ocr_pool:      ThreadPoolExecutor,
    mode:          str,
    chunk_size:    int,
    chunk_overlap: int,
) -> List[Tuple[int, int, List[Document]]]:
    """
    Convert one PDF segment and chunk its pages.

    Step A — Docling base conversion on the whole segment.
    Step B — Pages that fail is_encoding_garbage() are submitted to the
             persistent ocr_pool concurrently.
    Step C — Pages that pass post-OCR is_completely_unreadable() are chunked.

    Returns list of (seg_idx, global_page_num, documents).
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "subtitle"), ("###", "section")],
        strip_headers=False,
    )
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # ── Step A: Docling base conversion ─────────────────────────────────
    with suppress_internal_noise():
        result   = _get_base_converter().convert(seg_path)
        doc_data = result.document

    page_texts:  dict[int, str]                = {}
    ocr_futures: dict[Future, Tuple[int, int]] = {}

    for local_idx, _ in doc_data.pages.items():
        global_page_num = seg_start + local_idx
        page_md         = doc_data.export_to_markdown(page_no=local_idx)

        garbage, reason = is_encoding_garbage(page_md)
        if garbage:
            logger.info(
                f"[seg {seg_idx}] p.{global_page_num}: "
                f"OCR triggered ({reason})"
            )
            future = ocr_pool.submit(
                _ocr_single_page,
                seg_path,
                local_idx,
                global_page_num,
                seg_idx,
            )
            ocr_futures[future] = (global_page_num, local_idx)
        else:
            page_texts[global_page_num] = page_md

    # ── Step B: collect OCR results ──────────────────────────────────────
    for future in as_completed(ocr_futures):
        global_page_num, _ = ocr_futures[future]
        try:
            ocr_md = future.result()
            if ocr_md:
                page_texts[global_page_num] = ocr_md
            # empty string = is_completely_unreadable() already logged and skipped
        except Exception as exc:
            logger.error(
                f"[seg {seg_idx}] p.{global_page_num}: OCR future raised: {exc}"
            )

    # ── Step C: chunk all kept pages ─────────────────────────────────────
    results: List[Tuple[int, int, List[Document]]] = []

    for global_page_num, text in page_texts.items():
        if not text.strip():
            continue

        base_meta = {
            "source":      source,
            "filename":    os.path.basename(source),
            "page_number": global_page_num,
            "total_pages": num_pages,
        }

        if mode == "simple":
            docs = [Document(page_content=text, metadata=base_meta)]
        else:
            docs = []
            for chunk in md_splitter.split_text(text):
                chunk.metadata.update(base_meta)
                if len(chunk.page_content) > chunk_size:
                    docs.extend(rec_splitter.split_documents([chunk]))
                else:
                    docs.append(chunk)

        results.append((seg_idx, global_page_num, docs))

    return results


# ---------------------------------------------------------------------------
# PDF Processor
# ---------------------------------------------------------------------------

class AdvancedPDFProcessor:
    """
    Persistent-pool PDF processor with two-tier readability checks.

    Pools are created once in __init__ and warmed immediately so that
    the first extract_and_chunk() call has zero converter init cost.
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 300):
        try:
            self.chunk_size    = chunk_size    or CONF.get("CHUNK_SIZE",    1200)
            self.chunk_overlap = chunk_overlap or CONF.get("CHUNK_OVERLAP", 300)
        except Exception:
            self.chunk_size    = chunk_size    or 1200
            self.chunk_overlap = chunk_overlap or 300

        self.segment_workers = CONF.get("PDF_SEGMENT_WORKERS", 2)
        self.ocr_workers     = CONF.get("PDF_OCR_WORKERS",     2)

        # Persistent pools — never torn down between calls
        self.segment_pool = ThreadPoolExecutor(
            max_workers=self.segment_workers,
            thread_name_prefix="pdf_seg",
        )
        self.ocr_pool = ThreadPoolExecutor(
            max_workers=self.ocr_workers,
            thread_name_prefix="pdf_ocr",
        )

        self._warm_up_pools()

        logger.info(
            f"PDF Processor ready | "
            f"chunk_size={self.chunk_size} | "
            f"seg_workers={self.segment_workers} | "
            f"ocr_workers={self.ocr_workers}"
        )
        print(
            f"{Colors.BLUE}[INIT]{Colors.ENDC} PDF Processor ready | "
            f"ChunkSize={self.chunk_size} | "
            f"Workers: seg={self.segment_workers} ocr={self.ocr_workers} | "
            f"Engines: persistent (warm)",
            file=sys.stderr, flush=True,
        )

    def _warm_up_pools(self) -> None:
        """Initialise all thread-local converters now so first call is fast."""
        print(
            f"{Colors.YELLOW}[WARM-UP]{Colors.ENDC} Initialising engines…",
            file=sys.stderr, flush=True,
        )
        futures = (
            [self.segment_pool.submit(_get_base_converter)
             for _ in range(self.segment_workers)]
            +
            [self.ocr_pool.submit(_get_ocr_converter)
             for _ in range(self.ocr_workers)]
        )
        for f in futures:
            try:
                f.result()
            except Exception as e:
                logger.error(f"Warm-up job failed: {e}")
        print(
            f"{Colors.GREEN}[WARM-UP]{Colors.ENDC} "
            f"{self.segment_workers} base + {self.ocr_workers} OCR engines ready.",
            file=sys.stderr, flush=True,
        )

    def __del__(self) -> None:
        try:
            self.segment_pool.shutdown(wait=False)
            self.ocr_pool.shutdown(wait=False)
        except Exception:
            pass

    def _download_url(self, url: str) -> str:
        logger.info(f"Downloading PDF: {url}")
        try:
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                r = client.get(url)
                r.raise_for_status()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(r.content)
                tmp.close()
                return tmp.name
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return ""

    def _build_segments(self, num_pages: int) -> List[Tuple[int, int]]:
        segments, current = [], 0
        while current < num_pages:
            if num_pages - current <= 15:
                segments.append((current, num_pages))
                break
            segments.append((current, current + 10))
            current += 10
        return segments

    def extract_and_chunk(
        self,
        source: str,
        mode: Literal["simple", "advanced"] = "advanced",
    ) -> List[Document]:
        """
        Extract and chunk a PDF. Segments are processed concurrently.
        Results are sorted by (seg_idx, page_num) to preserve order.
        """
        target_path = source
        is_temp     = False
        t_start     = time.time()

        if source.lower().startswith(("http://", "https://")):
            target_path = self._download_url(source)
            if not target_path:
                return []
            is_temp = True

        if not os.path.exists(target_path):
            logger.error(f"File not found: {target_path}")
            return []

        try:
            reader    = pypdf.PdfReader(target_path)
            num_pages = len(reader.pages)
            segments  = self._build_segments(num_pages)

            print(
                f"{Colors.YELLOW}[START]{Colors.ENDC} "
                f"{os.path.basename(source)} | "
                f"{num_pages} pages | {len(segments)} segments",
                file=sys.stderr, flush=True,
            )
            logger.info(
                f"Processing: {os.path.basename(source)} | "
                f"{num_pages} pages | {len(segments)} segments"
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                # Write all segment files upfront (sequential, fast)
                seg_paths: List[str] = []
                for idx, (seg_start, seg_end) in enumerate(segments):
                    seg_path = os.path.join(temp_dir, f"seg_{idx}.pdf")
                    writer   = pypdf.PdfWriter()
                    for p in range(seg_start, seg_end):
                        writer.add_page(reader.pages[p])
                    with open(seg_path, "wb") as f:
                        writer.write(f)
                    seg_paths.append(seg_path)

                # Submit all segments to persistent pool
                all_raw:   List[Tuple[int, int, List[Document]]] = []
                done_count = [0]
                done_lock  = threading.Lock()

                future_to_idx = {
                    self.segment_pool.submit(
                        _process_segment,
                        idx,
                        segments[idx][0],
                        seg_paths[idx],
                        source,
                        num_pages,
                        self.ocr_pool,
                        mode,
                        self.chunk_size,
                        self.chunk_overlap,
                    ): idx
                    for idx in range(len(segments))
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        all_raw.extend(future.result())
                    except Exception as exc:
                        logger.error(f"Segment {idx} raised: {exc}", exc_info=True)
                    with done_lock:
                        done_count[0] += 1
                        n = done_count[0]
                    s, e = segments[idx]
                    print(
                        f"  {Colors.GREEN}✓{Colors.ENDC} "
                        f"Segment {n}/{len(segments)} (pp. {s+1}–{e})",
                        file=sys.stderr, flush=True,
                    )

            # Restore original page order
            all_raw.sort(key=lambda x: (x[0], x[1]))
            final_documents = [doc for _, _, docs in all_raw for doc in docs]

            elapsed = time.time() - t_start
            summary = (
                f"{len(final_documents)} chunks | "
                f"{elapsed:.1f}s (~{elapsed/num_pages:.2f}s/page)"
            )
            print(
                f"{Colors.GREEN}[DONE]{Colors.ENDC} {summary}",
                file=sys.stderr, flush=True,
            )
            logger.info(summary)
            return final_documents

        except Exception as e:
            logger.critical(f"Fatal error: {e}", exc_info=True)
            return []

        finally:
            if is_temp and os.path.exists(target_path):
                os.remove(target_path)
                logger.info("Temp file cleaned up.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aether PDF Processor")
    parser.add_argument("source",                                 help="PDF path or URL")
    parser.add_argument("--mode",            default="advanced",  choices=["simple", "advanced"])
    parser.add_argument("--chunk-size",      type=int)
    parser.add_argument("--segment-workers", type=int)
    parser.add_argument("--ocr-workers",     type=int)
    args = parser.parse_args()

    if args.segment_workers: CONF["PDF_SEGMENT_WORKERS"] = args.segment_workers
    if args.ocr_workers:     CONF["PDF_OCR_WORKERS"]     = args.ocr_workers

    processor = AdvancedPDFProcessor(
        chunk_size=args.chunk_size or CONF.get("CHUNK_SIZE", 1200),
    )
    docs = processor.extract_and_chunk(args.source, mode=args.mode)

    if not docs:
        print(f"\n{Colors.RED}No documents extracted.{Colors.ENDC}")
    else:
        print(f"\n{Colors.GREEN}Extracted {len(docs)} chunks.{Colors.ENDC}")
        print("\nFirst 3 chunks:\n")
        for i, doc in enumerate(docs[:3]):
            pg = doc.metadata.get("page_number", "?")
            print(f"[{i+1}] p.{pg} — {doc.page_content[:300]}\n{'─'*60}")