"""
AgroSight – Cross-Encoder Re-ranker
=====================================
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to rerank top-K retrieved chunks.
Adds ~50ms but improves precision by 15–20% on agriculture queries.

Strategy (from report):
  1. Retrieve top_k * 2 candidates from Qdrant (default 8 → 16)
  2. Cross-encode each (query, chunk) pair
  3. Return top rerank_top_k (default 5) by cross-encoder score
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.utils.logger import logger

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder  # type: ignore

    logger.info(f"Loading cross-encoder: {_RERANKER_MODEL}")
    model = CrossEncoder(_RERANKER_MODEL, max_length=512)
    logger.success("Cross-encoder loaded")
    return model


def preload_models():
    """Proactively load the cross-encoder model."""
    _get_cross_encoder()


# ---------------------------------------------------------------------------
# Public rerank function
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Rerank *candidates* (list of Qdrant payload dicts with 'text' key)
    against *query* using the cross-encoder.

    Returns at most *top_k* results sorted by descending relevance score.
    """
    if not candidates:
        return []

    try:
        cross_encoder = _get_cross_encoder()
    except Exception as exc:
        logger.warning(f"Cross-encoder unavailable ({exc}), skipping rerank")
        return candidates[:top_k]

    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, doc in ranked[:top_k]:
        doc = doc.copy()
        doc["rerank_score"] = float(score)
        results.append(doc)

    logger.debug(f"Reranked {len(candidates)} → {len(results)} results")
    return results
