"""
AgroSight – Full Ingestion Pipeline
=====================================
End-to-end script to:
  1. Walk all data directories
  2. Chunk each file using the hybrid auto-selector
  3. Embed chunks in batches (bge-m3)
  4. Upsert to Qdrant with SHA-256 deduplication

Usage:
    python -m scripts.ingest [--data-root ./data] [--recreate-collection]

The script is idempotent: re-running will skip already-ingested chunks
(matched by SHA-256 hash stored in Qdrant payload).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from app.services.chunker import ChunkResult, chunk_file
from app.services.embedder import encode_sparse, encode_texts
from app.services.vector_store import ensure_collection, upsert_chunks
from app.utils.config import get_settings
from app.utils.file_utils import iter_data_files
from app.utils.logger import configure_logger, logger

settings = get_settings()


def run_ingestion(
    data_root: str | Path,
    books_dir: str | Path,
    batch_size: int = 50,
    recreate_collection: bool = False,
) -> dict[str, int]:
    """
    Run the full ingestion pipeline.

    Returns:
        dict with total_files, total_chunks, total_upserted, total_skipped
    """
    configure_logger()

    # ── Ensure Qdrant collection ─────────────────────────────────────────
    logger.info(f"Ensuring Qdrant collection '{settings.qdrant_collection}' (dim={settings.embedding_dim})")
    ensure_collection(recreate=recreate_collection)

    # ── Collect all files ────────────────────────────────────────────────
    roots = [Path(data_root), Path(books_dir)]
    all_files: list[Path] = []
    for root in roots:
        if root.exists():
            all_files.extend(iter_data_files(root))
        else:
            logger.warning(f"Directory not found: {root}")

    logger.info(f"Found {len(all_files)} ingestible files")

    stats = {"total_files": len(all_files), "total_chunks": 0, "total_upserted": 0, "total_skipped": 0}

    # ── Process each file ────────────────────────────────────────────────
    pending_chunks: list[ChunkResult] = []

    for file_path in tqdm(all_files, desc="Chunking files", unit="file"):
        try:
            chunks = chunk_file(file_path)
        except Exception as exc:
            logger.error(f"Chunking failed for {file_path}: {exc}")
            continue

        if not chunks:
            continue

        stats["total_chunks"] += len(chunks)
        pending_chunks.extend(chunks)

        # Flush when enough chunks accumulate
        if len(pending_chunks) >= batch_size:
            upserted, skipped = _embed_and_upsert(pending_chunks)
            stats["total_upserted"] += upserted
            stats["total_skipped"] += skipped
            pending_chunks = []

    # Flush remainder
    if pending_chunks:
        upserted, skipped = _embed_and_upsert(pending_chunks)
        stats["total_upserted"] += upserted
        stats["total_skipped"] += skipped

    logger.success(
        f"Ingestion complete — "
        f"{stats['total_files']} files | "
        f"{stats['total_chunks']} chunks | "
        f"{stats['total_upserted']} upserted | "
        f"{stats['total_skipped']} skipped (already existed)"
    )
    return stats


def _embed_and_upsert(chunks: list[ChunkResult]) -> tuple[int, int]:
    """Embed a batch of chunks and upsert to Qdrant. Returns (upserted, skipped)."""
    texts = [c.text for c in chunks]

    # Dense embeddings
    try:
        vectors = encode_texts(texts, show_progress=False)
    except Exception as exc:
        logger.error(f"Embedding failed: {exc}")
        return 0, len(chunks)

    # Sparse weights (bge-m3 only)
    use_sparse = "bge-m3" in settings.embedding_model.lower()
    if use_sparse:
        try:
            sparse_weights = encode_sparse(texts)
        except Exception as exc:
            logger.warning(f"Sparse encoding failed: {exc}. Using dense only.")
            sparse_weights = [{} for _ in chunks]
    else:
        sparse_weights = [{} for _ in chunks]

    # Build Qdrant payload dicts
    chunk_dicts: list[dict[str, Any]] = []
    for chunk, vec, sparse in zip(chunks, vectors, sparse_weights):
        chunk_dicts.append(
            {
                "vector": vec.tolist(),
                "text": chunk.text,
                "chunk_hash": chunk.chunk_hash,
                "source_file": chunk.source_file,
                "chunk_type": chunk.chunk_type,
                "chunk_index": chunk.chunk_index,
                "crop_category": chunk.crop_category,
                "language": chunk.language,
                "metadata": chunk.metadata,
                "sparse_vector": sparse,
            }
        )

    upserted = upsert_chunks(chunk_dicts, skip_existing=True)
    skipped = len(chunks) - upserted
    return upserted, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroSight Ingestion Pipeline")
    parser.add_argument("--data-root", default=settings.data_root, help="Path to data/raw")
    parser.add_argument("--books-dir", default=settings.books_dir, help="Path to data/books")
    parser.add_argument("--batch-size", type=int, default=50, help="Chunks per embedding batch")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="DROP and recreate Qdrant collection (Phase 5 migration)",
    )
    args = parser.parse_args()

    run_ingestion(
        data_root=args.data_root,
        books_dir=args.books_dir,
        batch_size=args.batch_size,
        recreate_collection=args.recreate_collection,
    )
