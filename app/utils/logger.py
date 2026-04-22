"""
AgroSight – Centralised structured logger (loguru).
Import `logger` everywhere instead of calling logging.getLogger().
"""

import sys

from loguru import logger

from app.utils.config import get_settings


def configure_logger() -> None:
    """Configure loguru based on settings. Call once at app startup."""
    settings = get_settings()
    logger.remove()  # Remove default stderr handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> – "
            "<level>{message}</level>"
        ),
    )
    logger.add(
        "logs/agrosight.log",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
    )


__all__ = ["logger", "configure_logger"]
