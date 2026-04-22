"""
AgroSight – File-system helpers used by the ingestion pipeline.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Generator

# Extensions that are never text-chunked (go to vision pipeline instead)
VISION_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
)

# Extensions we can attempt to ingest
INGESTIBLE_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".json", ".csv", ".parquet", ".html", ".txt", ".md"}
)


def sha256_of_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_text(text: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_data_files(
    root: str | Path,
    *,
    recursive: bool = True,
    skip_vision: bool = True,
) -> Generator[Path, None, None]:
    """
    Yield every ingestible file under *root*.

    Args:
        root: Base directory to walk.
        recursive: Walk sub-directories if True.
        skip_vision: Skip image files (they go to vision pipeline).
    """
    root = Path(root)
    pattern = "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if skip_vision and ext in VISION_EXTENSIONS:
            continue
        if ext in INGESTIBLE_EXTENSIONS:
            yield p


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if they don't exist; return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def relative_to_data_root(path: str | Path, data_root: str | Path) -> str:
    """Return *path* relative to *data_root* as a POSIX string."""
    try:
        return Path(path).relative_to(data_root).as_posix()
    except ValueError:
        return str(path)
