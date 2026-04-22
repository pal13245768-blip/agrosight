"""
AgroSight – Qdrant Vector Store Service
=========================================
Handles:
  • Collection creation / recreation (dim-aware)
  • Upsert with deduplication via SHA-256 payload filter
  • Dense cosine search (Phase 3)
  • Hybrid dense + BM25 sparse search (Phase 4+, bge-m3)
  • Reciprocal Rank Fusion for hybrid merging
  • Payload-based filtering (crop_category, chunk_type, language)
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.utils.config import get_settings
from app.utils.logger import logger

settings = get_settings()

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.request_timeout,
        )
        logger.info(f"Qdrant client connected: {settings.qdrant_url}")
    return _client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


def ensure_collection(
    collection_name: str | None = None,
    dim: int | None = None,
    recreate: bool = False,
) -> None:
    """
    Create collection if it doesn't exist.
    If *recreate* is True, drop and recreate (use only for Phase 5 migration).
    """
    collection_name = collection_name or settings.qdrant_collection
    dim = dim or settings.embedding_dim
    client = get_client()

    exists = False
    try:
        info = client.get_collection(collection_name)
        existing_dim = info.config.params.vectors.size
        exists = True
        if existing_dim != dim:
            logger.warning(
                f"Collection '{collection_name}' exists with dim={existing_dim}, "
                f"requested dim={dim}. Use recreate=True to migrate."
            )
            return
    except Exception:
        exists = False

    if exists and not recreate:
        logger.info(f"Collection '{collection_name}' already exists – skipping creation")
        return

    if exists and recreate:
        logger.warning(f"Recreating collection '{collection_name}' (all data will be lost!)")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
            on_disk=False,
        ),
        # Sparse vectors for bge-m3 hybrid search (Phase 4+)
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
            )
        } if dim == 1024 else None,
    )
    logger.success(f"Collection '{collection_name}' created (dim={dim})")


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def upsert_chunks(
    chunks_data: list[dict[str, Any]],
    collection_name: str | None = None,
    skip_existing: bool = True,
) -> int:
    """
    Upsert a list of chunk dicts into Qdrant.

    Each dict must have:
      - 'vector'     : list[float]  (dense embedding)
      - 'text'       : str
      - 'chunk_hash' : str  (SHA-256, used for dedup)
      - 'source_file': str
      - 'chunk_type' : str
      - 'chunk_index': int
      - 'crop_category': str
      - 'language'   : str
      - 'sparse_vector': dict[str, float]  (optional, Phase 4+)
      - 'metadata'   : dict  (optional extra payload)

    Returns number of points actually upserted.
    """
    collection_name = collection_name or settings.qdrant_collection
    client = get_client()
    upserted = 0

    points: list[models.PointStruct] = []

    for chunk in chunks_data:
        chunk_hash = chunk["chunk_hash"]

        # Deduplication: check if hash already in collection
        if skip_existing:
            hits, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_hash",
                            match=models.MatchValue(value=chunk_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            if hits:
                continue

        payload = {
            "text": chunk["text"],
            "chunk_hash": chunk_hash,
            "source_file": chunk.get("source_file", ""),
            "chunk_type": chunk.get("chunk_type", "text"),
            "chunk_index": chunk.get("chunk_index", 0),
            "crop_category": chunk.get("crop_category", ""),
            "language": chunk.get("language", "en"),
            **(chunk.get("metadata") or {}),
        }

        # Build sparse vector if provided (for hybrid collection)
        sparse = chunk.get("sparse_vector") or {}
        named_vectors: dict[str, Any] = {}

        if sparse and settings.embedding_dim == 1024:
            named_vectors["bm25"] = models.SparseVector(
                indices=list(sparse.keys()),
                values=list(sparse.values()),
            )

        if named_vectors:
            # Hybrid point with both dense and sparse
            pt = models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"": chunk["vector"], **named_vectors},
                payload=payload,
            )
        else:
            pt = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk["vector"],
                payload=payload,
            )

        points.append(pt)

        if len(points) >= settings.batch_size:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            upserted += len(points)
            logger.debug(f"Upserted batch of {len(points)} points")
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)
        upserted += len(points)

    return upserted


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------


def dense_search(
    query_vector: np.ndarray | list[float],
    top_k: int | None = None,
    score_threshold: float | None = None,
    collection_name: str | None = None,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Cosine similarity search. Returns list of {score, payload} dicts."""
    collection_name = collection_name or settings.qdrant_collection
    top_k = top_k or settings.retrieval_top_k
    score_threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold
    client = get_client()

    qf = _build_filter(filters) if filters else None  # _build_filter may also return None

    hits = client.search(
        collection_name=collection_name,
        query_vector=list(query_vector) if isinstance(query_vector, np.ndarray) else query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=qf,
        with_payload=True,
    )
    return [{"score": h.score, **h.payload} for h in hits]


# ---------------------------------------------------------------------------
# Sparse (BM25) retrieval — Phase 4+
# ---------------------------------------------------------------------------


def sparse_search(
    sparse_weights: dict[str, float],
    top_k: int | None = None,
    collection_name: str | None = None,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """BM25-style sparse search using bge-m3 lexical weights."""
    collection_name = collection_name or settings.qdrant_collection
    top_k = top_k or settings.retrieval_top_k
    client = get_client()

    if not sparse_weights:
        return []

    qf = _build_filter(filters) if filters else None

    try:
        hits = client.search(
            collection_name=collection_name,
            query_vector=models.NamedSparseVector(
                name="bm25",
                vector=models.SparseVector(
                    indices=list(sparse_weights.keys()),
                    values=list(sparse_weights.values()),
                ),
            ),
            limit=top_k,
            query_filter=qf,
            with_payload=True,
        )
        return [{"score": h.score, **h.payload} for h in hits]
    except Exception as exc:
        logger.warning(f"Sparse search failed: {exc}. Falling back to dense-only.")
        return []


# ---------------------------------------------------------------------------
# Hybrid retrieval (dense + sparse, merged via RRF)
# ---------------------------------------------------------------------------


def hybrid_search(
    query_vector: np.ndarray | list[float],
    sparse_weights: dict[str, float] | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
    collection_name: str | None = None,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Phase 4+ hybrid retrieval: dense cosine + BM25 sparse merged by RRF.
    Falls back to dense-only if sparse weights not provided.
    """
    top_k = top_k or settings.retrieval_top_k

    dense_results = dense_search(
        query_vector,
        top_k=top_k * 2,  # retrieve more candidates for RRF
        score_threshold=0.0,  # threshold applied post-fusion
        collection_name=collection_name,
        filters=filters,
    )

    if sparse_weights:
        sparse_results = sparse_search(
            sparse_weights,
            top_k=top_k * 2,
            collection_name=collection_name,
            filters=filters,
        )
    else:
        sparse_results = []

    merged = _rrf_merge(dense_results, sparse_results, top_k=top_k)

    # Apply score threshold post-fusion
    threshold = score_threshold if score_threshold is not None else settings.retrieval_score_threshold
    return [r for r in merged if r.get("rrf_score", 0) >= threshold * 0.5]  # RRF scores differ from cosine


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _rrf_merge(
    dense: list[dict],
    sparse: list[dict],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, doc in enumerate(dense):
        key = doc.get("chunk_hash", str(rank))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        docs[key] = doc

    for rank, doc in enumerate(sparse):
        key = doc.get("chunk_hash", str(rank))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        docs[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    results = []
    for key in sorted_keys:
        d = docs[key].copy()
        d["rrf_score"] = scores[key]
        results.append(d)
    return results


# ---------------------------------------------------------------------------
# Filter builder
# ---------------------------------------------------------------------------


def _build_filter(filters: dict[str, Any]) -> models.Filter | None:
    """Convert a simple dict of key=value to a Qdrant must-filter.

    Skips entries whose value is None, an empty collection, or a non-scalar
    type (e.g. a nested dict) that Qdrant's MatchValue cannot accept.
    Returns None when no valid conditions remain.
    """
    conditions = []
    for key, value in filters.items():
        # Skip None or empty values
        if value is None or value == {} or value == []:
            continue
        if isinstance(value, list):
            # Filter out non-scalar items inside the list
            scalar_items = [v for v in value if isinstance(v, (str, int, float, bool))]
            if scalar_items:
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=scalar_items))
                )
        elif isinstance(value, (str, int, float, bool)):
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
        else:
            logger.warning(f"Skipping unsupported filter value for key '{key}': {type(value).__name__}")
    if not conditions:
        return None
    return models.Filter(must=conditions)
