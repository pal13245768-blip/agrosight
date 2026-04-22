"""
AgroSight – Session Store (Redis)
===================================
Manages 5-turn conversation history per session_id.
Falls back to an in-memory dict if Redis is unavailable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from app.utils.config import get_settings
from app.utils.logger import logger

settings = get_settings()

# ---------------------------------------------------------------------------
# Redis client (optional)
# ---------------------------------------------------------------------------

_redis = None


def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis  # type: ignore

        r = redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=2)
        r.ping()
        _redis = r
        logger.info("Redis session store connected")
    except Exception as exc:
        logger.warning(f"Redis unavailable ({exc}). Using in-memory session store.")
        _redis = None
    return _redis


# In-memory fallback
_memory_store: dict[str, list[dict]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_history(session_id: str) -> list[dict[str, str]]:
    """Return conversation history list for *session_id*."""
    r = _get_redis()
    if r is not None:
        raw = r.get(f"session:{session_id}")
        if raw:
            return json.loads(raw)
        return []
    return list(_memory_store[session_id])


def append_turn(session_id: str, role: str, content: str) -> None:
    """Append a new {role, content} turn and enforce MAX_HISTORY_TURNS."""
    history = get_history(session_id)
    history.append({"role": role, "content": content})
    # Keep only last N turns (user + assistant pairs)
    max_turns = settings.max_history_turns * 2  # each turn = user + assistant
    if len(history) > max_turns:
        history = history[-max_turns:]

    r = _get_redis()
    if r is not None:
        r.setex(
            f"session:{session_id}",
            settings.session_ttl_seconds,
            json.dumps(history),
        )
    else:
        _memory_store[session_id] = history


def clear_session(session_id: str) -> None:
    """Delete session history."""
    r = _get_redis()
    if r is not None:
        r.delete(f"session:{session_id}")
    else:
        _memory_store.pop(session_id, None)
