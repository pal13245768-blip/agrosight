"""
AgroSight – Chunking Engine
============================
Implements every chunking strategy described in the RAG Pipeline Strategy Report:
  • Semantic (sentence-boundary)       – flowing prose PDFs
  • Section-based (header detection)   – structured PDFs with numbered headings
  • Q&A pair                           – FAQ JSON / CSV
  • Table / row-narrative              – market & soil CSV
  • Record narrative                   – scheme / disease / soil JSON
  • Sliding window (fallback)          – unstructured / mixed files
  • Hybrid auto-select                 – routes each file to the right strategy

Each strategy returns a list of ChunkResult dataclass instances.
The auto-selector is the single public entry-point for the ingestion pipeline.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
from pypdf import PdfReader

from app.utils.config import get_settings
from app.utils.file_utils import sha256_of_text
from app.utils.logger import logger

settings = get_settings()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChunkResult:
    """A single text chunk ready for embedding and Qdrant upsert."""

    text: str
    chunk_hash: str = field(init=False)
    source_file: str = ""
    chunk_type: str = "text"           # text | qa | price_summary | nutrient_record | etc.
    chunk_index: int = 0
    crop_category: str = ""
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chunk_hash = sha256_of_text(self.text)


# ---------------------------------------------------------------------------
# Helper: sentence splitter (simple, no NLTK dependency)
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _words(text: str) -> list[str]:
    return text.split()


# ---------------------------------------------------------------------------
# Strategy 1 – Semantic chunking (sentence boundary)
# ---------------------------------------------------------------------------


def semantic_chunks(
    text: str,
    source_file: str = "",
    chunk_size: int | None = None,
    overlap: int | None = None,
    chunk_type: str = "text",
    crop_category: str = "",
    extra_meta: dict | None = None,
) -> list[ChunkResult]:
    """Split *text* at sentence boundaries into ~chunk_size word chunks."""
    chunk_size = chunk_size or settings.default_chunk_tokens
    overlap = overlap or settings.default_overlap_tokens
    sentences = _split_sentences(text)
    chunks: list[ChunkResult] = []
    current: list[str] = []
    current_words = 0
    idx = 0

    for sent in sentences:
        sw = len(_words(sent))
        if current_words + sw > chunk_size and current:
            chunk_text = " ".join(current)
            chunks.append(
                ChunkResult(
                    text=chunk_text,
                    source_file=source_file,
                    chunk_type=chunk_type,
                    chunk_index=idx,
                    crop_category=crop_category,
                    metadata=extra_meta or {},
                )
            )
            idx += 1
            # Keep overlap sentences
            overlap_sents: list[str] = []
            overlap_w = 0
            for s in reversed(current):
                w = len(_words(s))
                if overlap_w + w <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_w += w
                else:
                    break
            current = overlap_sents
            current_words = overlap_w
        current.append(sent)
        current_words += sw

    if current:
        chunks.append(
            ChunkResult(
                text=" ".join(current),
                source_file=source_file,
                chunk_type=chunk_type,
                chunk_index=idx,
                crop_category=crop_category,
                metadata=extra_meta or {},
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2 – Section-based chunking (header detection)
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(
    r'^(?:'
    r'\d+(?:\.\d+)*[\s\.]+'       # 1. / 1.2. / 1.2.3
    r'|Chapter\s+\d+'
    r'|Section\s+\d+'
    r'|CHAPTER\s+\d+'
    r'|[A-Z][A-Z\s]{4,}$'         # ALL CAPS headings ≥ 5 chars
    r')',
    re.MULTILINE | re.IGNORECASE,
)


def section_chunks(
    text: str,
    source_file: str = "",
    chunk_size: int | None = None,
    overlap: int | None = None,
    crop_category: str = "",
) -> list[ChunkResult]:
    """Split at detected section headers; fall back to semantic if section is too large."""
    chunk_size = chunk_size or settings.default_chunk_tokens
    overlap = overlap or settings.default_overlap_tokens
    lines = text.splitlines(keepends=True)
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if _HEADER_RE.match(line.strip()) and current:
            sections.append("".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("".join(current).strip())

    chunks: list[ChunkResult] = []
    for sec in sections:
        if not sec:
            continue
        if len(_words(sec)) <= chunk_size:
            chunks.append(
                ChunkResult(
                    text=sec,
                    source_file=source_file,
                    chunk_type="text",
                    chunk_index=len(chunks),
                    crop_category=crop_category,
                )
            )
        else:
            sub = semantic_chunks(
                sec, source_file=source_file, chunk_size=chunk_size,
                overlap=overlap, crop_category=crop_category,
            )
            for c in sub:
                c.chunk_index = len(chunks)
                chunks.append(c)
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3 – Q&A pair chunks
# ---------------------------------------------------------------------------


def qa_pair_chunks(
    records: list[dict],
    source_file: str = "",
    q_key: str = "question",
    a_key: str = "answer",
    crop_category: str = "",
) -> list[ChunkResult]:
    """One chunk per Q&A pair. Embeds question + answer together."""
    chunks: list[ChunkResult] = []
    for i, rec in enumerate(records):
        q = rec.get(q_key, rec.get("Q", "")).strip()
        a = rec.get(a_key, rec.get("A", "")).strip()
        if not q or not a:
            continue
        text = f"Q: {q}\nA: {a}"
        chunks.append(
            ChunkResult(
                text=text,
                source_file=source_file,
                chunk_type="qa",
                chunk_index=i,
                crop_category=crop_category,
                metadata={"question": q},
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 4 – Table / row-narrative chunks (CSV)
# ---------------------------------------------------------------------------


def table_row_chunks(
    df: pd.DataFrame,
    source_file: str = "",
    group_by: str | None = None,
    batch_size: int = 10,
    chunk_type: str = "price_summary",
    crop_category: str = "",
) -> list[ChunkResult]:
    """
    Convert tabular data to natural-language chunks.
    If *group_by* column exists, produce one chunk per group (e.g. per commodity).
    Otherwise batch rows in groups of *batch_size*.
    """
    chunks: list[ChunkResult] = []
    idx = 0

    if group_by and group_by in df.columns:
        for group_val, gdf in df.groupby(group_by):
            rows_text = _df_to_narrative(gdf)
            text = f"Commodity / Group: {group_val}\n{rows_text}"
            chunks.append(
                ChunkResult(
                    text=text,
                    source_file=source_file,
                    chunk_type=chunk_type,
                    chunk_index=idx,
                    crop_category=crop_category,
                    metadata={"group": str(group_val)},
                )
            )
            idx += 1
    else:
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]
            text = _df_to_narrative(batch)
            chunks.append(
                ChunkResult(
                    text=text,
                    source_file=source_file,
                    chunk_type=chunk_type,
                    chunk_index=idx,
                    crop_category=crop_category,
                )
            )
            idx += 1

    return chunks


def _df_to_narrative(df: pd.DataFrame) -> str:
    """Convert a small DataFrame to a compact natural-language paragraph."""
    lines: list[str] = []
    for _, row in df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        lines.append("; ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strategy 5 – Record narrative chunks (JSON objects)
# ---------------------------------------------------------------------------


def record_narrative_chunks(
    records: list[dict],
    source_file: str = "",
    chunk_type: str = "scheme_record",
    crop_category: str = "",
    title_key: str | None = None,
) -> list[ChunkResult]:
    """One chunk per JSON record. Converts nested dict to readable narrative."""
    chunks: list[ChunkResult] = []
    for i, rec in enumerate(records):
        title = rec.get(title_key, "") if title_key else ""
        narrative = _dict_to_narrative(rec, indent=0)
        text = f"{title}\n{narrative}".strip() if title else narrative
        chunks.append(
            ChunkResult(
                text=text,
                source_file=source_file,
                chunk_type=chunk_type,
                chunk_index=i,
                crop_category=crop_category,
                metadata={"title": str(title)},
            )
        )
    return chunks


def _dict_to_narrative(obj: Any, indent: int = 0, max_depth: int = 4) -> str:
    """Recursively flatten a dict/list to a readable string."""
    if indent > max_depth:
        return str(obj)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            label = str(k).replace("_", " ").title()
            if isinstance(v, (dict, list)):
                parts.append(f"{label}:\n{_dict_to_narrative(v, indent+1, max_depth)}")
            else:
                parts.append(f"{label}: {v}")
        return "\n".join(parts)
    if isinstance(obj, list):
        if all(isinstance(i, str) for i in obj):
            return "; ".join(obj)
        return "\n".join(_dict_to_narrative(i, indent, max_depth) for i in obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Strategy 6 – Sliding window (fallback)
# ---------------------------------------------------------------------------


def sliding_window_chunks(
    text: str,
    source_file: str = "",
    chunk_size: int | None = None,
    overlap: int | None = None,
    crop_category: str = "",
) -> list[ChunkResult]:
    """Fixed-size sliding window over word tokens (pure fallback)."""
    chunk_size = chunk_size or settings.default_chunk_tokens
    overlap = overlap or settings.default_overlap_tokens
    words = _words(text)
    chunks: list[ChunkResult] = []
    step = max(1, chunk_size - overlap)
    for i, start in enumerate(range(0, len(words), step)):
        segment = words[start : start + chunk_size]
        if not segment:
            break
        chunks.append(
            ChunkResult(
                text=" ".join(segment),
                source_file=source_file,
                chunk_type="text",
                chunk_index=i,
                crop_category=crop_category,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# PDF text extraction helpers
# ---------------------------------------------------------------------------


def _extract_pdf_text(path: Path) -> str:
    """
    Extract full text from a PDF.
    Priority: pdfplumber → pypdf → PyMuPDF (scanned pages via OCR flag).
    """
    text_parts: list[str] = []

    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
        if text_parts:
            return "\n".join(text_parts)
    except Exception as exc:
        logger.warning(f"pdfplumber failed for {path}: {exc}")

    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except Exception as exc:
        logger.warning(f"pypdf failed for {path}: {exc}")

    # PyMuPDF last resort
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        for page in doc:
            text_parts.append(page.get_text())
        return "\n".join(text_parts)
    except Exception as exc:
        logger.error(f"PyMuPDF failed for {path}: {exc}")

    return ""


# ---------------------------------------------------------------------------
# Auto-selector – maps each file to the correct strategy
# ---------------------------------------------------------------------------


def _detect_csv_group_column(df: pd.DataFrame) -> str | None:
    """Try to detect a commodity / crop column for grouping."""
    candidates = [
        "Commodity", "commodity", "Crop", "crop", "crop_name",
        "COMMODITY", "CROP", "Variety", "variety",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _classify_csv(path: Path, df: pd.DataFrame) -> str:
    """Return chunk_type label for a CSV file."""
    stem = path.stem.lower()
    if any(k in stem for k in ("price", "mandi", "agmarket", "market")):
        return "price_summary"
    if any(k in stem for k in ("nutrient", "fertilizer", "fertiliser")):
        return "nutrient_record"
    if any(k in stem for k in ("soil", "soilgrids")):
        return "soil_classification"
    if any(k in stem for k in ("usda", "nass", "fao", "crop_stat")):
        return "crop_statistics"
    if "question" in [c.lower() for c in df.columns]:
        return "qa"
    return "tabular"


def _classify_json(path: Path) -> str:
    """Return chunk_type label for a JSON file."""
    stem = path.stem.lower()
    if any(k in stem for k in ("faq",)):
        return "qa"
    if any(k in stem for k in ("scheme", "all_scheme")):
        return "scheme_record"
    if any(k in stem for k in ("disease", "pest")):
        return "disease_record"
    if any(k in stem for k in ("soil",)):
        return "soil_classification"
    if any(k in stem for k in ("advisory",)):
        return "advisory"
    if any(k in stem for k in ("weather",)):
        return "weather"
    return "json_record"


def _infer_crop_category(path: Path) -> str:
    """Infer crop_category from directory path."""
    parts = [p.lower() for p in path.parts]
    if "government" in parts or "scheme" in parts:
        return "government"
    if "fertilizer" in parts or "fertiliser" in parts:
        return "fertilizer"
    if "soil" in parts:
        return "soil"
    if "weather" in parts:
        return "weather"
    if "pest" in parts or "disease" in parts:
        return "pest_disease"
    if "market" in parts or "mandi" in parts or "price" in parts:
        return "market"
    if "books" in parts:
        return "agronomy_book"
    return "general"


def chunk_file(path: str | Path) -> list[ChunkResult]:
    """
    Public entry-point for the ingestion pipeline.

    Given any supported file path, auto-detect the best chunking strategy
    and return a list of ChunkResult instances ready for embedding.

    Returns [] for vision files and unsupported extensions.
    """
    path = Path(path)
    ext = path.suffix.lower()
    source_file = str(path)
    crop_category = _infer_crop_category(path)

    # ── Vision files – skip ──────────────────────────────────────────────
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}:
        logger.debug(f"Skipping vision file: {path.name}")
        return []

    # ── PDF ──────────────────────────────────────────────────────────────
    if ext == ".pdf":
        logger.info(f"Chunking PDF: {path.name}")
        text = _extract_pdf_text(path)
        if not text.strip():
            logger.warning(f"No text extracted from {path.name}")
            return []
        # Choose section-based if headers detected, else semantic
        header_hits = len(_HEADER_RE.findall(text[:5000]))
        if header_hits >= 3:
            return section_chunks(text, source_file=source_file, crop_category=crop_category)
        return semantic_chunks(text, source_file=source_file, crop_category=crop_category)

    # ── CSV ──────────────────────────────────────────────────────────────
    if ext == ".csv":
        logger.info(f"Chunking CSV: {path.name}")
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logger.error(f"CSV read error {path.name}: {exc}")
            return []
        ctype = _classify_csv(path, df)
        if ctype == "qa":
            records = df.to_dict(orient="records")
            return qa_pair_chunks(records, source_file=source_file, crop_category=crop_category)
        group_col = _detect_csv_group_column(df)
        return table_row_chunks(
            df, source_file=source_file,
            group_by=group_col, chunk_type=ctype, crop_category=crop_category,
        )

    # ── Parquet ──────────────────────────────────────────────────────────
    if ext == ".parquet":
        logger.info(f"Chunking Parquet: {path.name}")
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.error(f"Parquet read error {path.name}: {exc}")
            return []
        group_col = _detect_csv_group_column(df)
        return table_row_chunks(
            df, source_file=source_file,
            group_by=group_col, chunk_type="tabular", crop_category=crop_category,
        )

    # ── JSON ─────────────────────────────────────────────────────────────
    if ext == ".json":
        logger.info(f"Chunking JSON: {path.name}")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.error(f"JSON read error {path.name}: {exc}")
            return []

        ctype = _classify_json(path)
        stem = path.stem.lower()

        # Normalise to list of records
        if isinstance(data, dict):
            # Try common list wrapper keys
            for key in ("data", "records", "items", "results", "schemes", "faqs"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                data = [data]

        if not isinstance(data, list):
            data = [data]

        if ctype == "qa":
            return qa_pair_chunks(data, source_file=source_file, crop_category=crop_category)

        # Title keys per type
        title_keys = {
            "scheme_record": "scheme_name",
            "disease_record": "disease_name",
            "soil_classification": "soil_type",
            "advisory": "location",
            "weather": "location",
        }
        return record_narrative_chunks(
            data, source_file=source_file,
            chunk_type=ctype, crop_category=crop_category,
            title_key=title_keys.get(ctype),
        )

    # ── HTML / TXT / MD ──────────────────────────────────────────────────
    if ext in {".html", ".htm", ".txt", ".md"}:
        logger.info(f"Chunking text file: {path.name}")
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.error(f"Text read error {path.name}: {exc}")
            return []
        if ext in {".html", ".htm"}:
            # Strip tags
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        return semantic_chunks(text, source_file=source_file, crop_category=crop_category)

    logger.debug(f"Unsupported extension {ext} for {path.name} – skipping")
    return []
