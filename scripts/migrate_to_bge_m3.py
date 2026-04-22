"""
AgroSight – Phase 5 Migration Script
========================================
Migrates from 384-dim collection (MiniLM) to 1024-dim collection (bge-m3).

Steps:
  1. Create new collection 'agricultural_knowledge_v2' (1024d)
  2. Re-ingest all data with bge-m3 embeddings
  3. Validate chunk counts
  4. (Optional) delete old collection after validation

Run AFTER updating EMBEDDING_MODEL and EMBEDDING_DIM in .env to bge-m3 / 1024.

Usage:
    python -m scripts.migrate_to_bge_m3 [--old-collection agricultural_knowledge] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.vector_store import ensure_collection, get_client
from app.utils.config import get_settings
from app.utils.logger import configure_logger, logger
from scripts.ingest import run_ingestion

configure_logger()
settings = get_settings()


def migrate(old_collection: str, dry_run: bool = False) -> None:
    """Perform Phase 5 migration."""
    new_collection = settings.qdrant_collection
    client = get_client()

    logger.info(f"Phase 5 Migration: {old_collection} → {new_collection}")
    logger.info(f"Embedding model: {settings.embedding_model} ({settings.embedding_dim}d)")

    if dry_run:
        logger.warning("[DRY RUN] No changes will be made.")
        return

    # Validate new collection does not already exist with wrong dim
    logger.info(f"Creating new collection '{new_collection}' (1024d)…")
    ensure_collection(collection_name=new_collection, dim=1024, recreate=False)

    # Run full ingestion into new collection
    logger.info("Starting re-ingestion into new collection…")
    stats = run_ingestion(
        data_root=settings.data_root,
        books_dir=settings.books_dir,
        batch_size=50,
        recreate_collection=False,  # collection already created above
    )

    logger.success(f"Re-ingestion complete: {stats}")

    # Validate
    new_count = client.count(collection_name=new_collection).count
    logger.success(f"New collection '{new_collection}' contains {new_count} vectors")

    if new_count > 0:
        logger.success("Migration validated. You can now:")
        logger.success(f"  1. Update DNS / load balancer to use '{new_collection}'")
        logger.success(f"  2. Keep '{old_collection}' as rollback for 24–48h")
        logger.success(f"  3. Delete '{old_collection}' after confirming production stability")
    else:
        logger.error("Migration FAILED – new collection is empty. Investigate before switching traffic.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroSight Phase 5 Migration")
    parser.add_argument("--old-collection", default="agricultural_knowledge", help="Old 384d collection name")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    args = parser.parse_args()
    migrate(old_collection=args.old_collection, dry_run=args.dry_run)
