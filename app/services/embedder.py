"""
AgroSight – Embedding Service
==============================
Wraps sentence-transformers / FlagEmbedding models with:
  • Lazy singleton loading (one model per process)
  • Batched encoding with progress bar
  • Normalised vectors (cosine similarity ready)
  • Transparent fallback from bge-m3 → MiniLM if hardware can't load bge-m3

Supported models (from strategy report):
  Phase 3  : all-MiniLM-L6-v2               (384d, English)
  Phase 3→4: multi-qa-MiniLM-L6-cos-v1      (384d, Q&A-trained, drop-in)
  Phase 4  : paraphrase-multilingual-MiniLM-L12-v2  (384d, multilingual)
  Phase 5  : BAAI/bge-m3                    (1024d, multilingual, RECOMMENDED)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from app.utils.config import get_settings
from app.utils.logger import logger

if TYPE_CHECKING:
    pass

settings = get_settings()

# ---------------------------------------------------------------------------
# Model loader with lazy singleton
# ---------------------------------------------------------------------------

_model = None          # sentence-transformers SentenceTransformer or BGEM3FlagModel
_model_name: str = ""  # track which model is loaded


def _load_model(model_name: str):
    """Load embedding model. Tries FlagEmbedding for bge-m3, else sentence-transformers."""
    global _model, _model_name

    if _model is not None and _model_name == model_name:
        return _model

    logger.info(f"Loading embedding model: {model_name}")

    if "bge-m3" in model_name.lower():
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore

            _model = BGEM3FlagModel(model_name, use_fp16=False)
            _model_name = model_name
            logger.success(f"bge-m3 loaded via FlagEmbedding ({settings.embedding_dim}d)")
            return _model
        except Exception as exc:
            logger.warning(f"FlagEmbedding load failed: {exc}. Falling back to sentence-transformers.")

    # sentence-transformers for all other models
    from sentence_transformers import SentenceTransformer  # type: ignore

    _model = SentenceTransformer(model_name)
    _model_name = model_name
    logger.success(f"Model loaded via sentence-transformers ({model_name})")
    return _model


def preload_models():
    """Proactively load the configured embedding model."""
    _load_model(settings.embedding_model)


# ---------------------------------------------------------------------------
# Core encoding function
# ---------------------------------------------------------------------------


def encode_texts(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode a list of texts into L2-normalised embedding vectors.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    model_name = model_name or settings.embedding_model
    model = _load_model(model_name)

    if not texts:
        return np.empty((0, settings.embedding_dim), dtype=np.float32)

    all_embeddings: list[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding", unit="batch")

    for start in iterator:
        batch = texts[start : start + batch_size]

        if "bge-m3" in model_name.lower():
            # FlagEmbedding API: returns dict with 'dense_vecs'
            output = model.encode(
                batch,
                batch_size=batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=False,   # enable True for Phase 4 hybrid
                return_colbert_vecs=False,
            )
            vecs = np.array(output["dense_vecs"], dtype=np.float32)
        else:
            # sentence-transformers API
            vecs = model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        # Ensure float32 and L2-normalised
        vecs = vecs.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms

        all_embeddings.append(vecs)

    return np.vstack(all_embeddings)


def encode_query(query: str, model_name: str | None = None) -> np.ndarray:
    """Encode a single query string. Returns shape (embedding_dim,)."""
    result = encode_texts([query], model_name=model_name)
    return result[0]


# ---------------------------------------------------------------------------
# Sparse encoding for bge-m3 hybrid search (Phase 4+)
# ---------------------------------------------------------------------------


def encode_sparse(texts: list[str]) -> list[dict[str, float]]:
    """
    Return sparse lexical weights from bge-m3 (for BM25-style hybrid retrieval).
    Falls back to empty dicts if not using bge-m3.
    """
    model_name = settings.embedding_model
    if "bge-m3" not in model_name.lower():
        return [{} for _ in texts]

    model = _load_model(model_name)
    try:
        output = model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        weights = output.get("lexical_weights", [{} for _ in texts])
        return [dict(w) for w in weights]
    except Exception as exc:
        logger.warning(f"Sparse encoding failed: {exc}")
        return [{} for _ in texts]
