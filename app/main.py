"""
AgroSight – FastAPI Application
=================================
Endpoints:
  GET  /health          – liveness check
  POST /search          – raw retrieval (no LLM)
  POST /chat            – SSE streaming chat with full RAG + ReAct agent
  DELETE /session/{id}  – clear conversation history

All endpoints are async and production-ready.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.services import embedder, reranker
from app.services.agent import retrieve_context, run_agent, stream_agent
from app.services.session_store import clear_session
from app.services.vector_store import get_client
from app.utils.config import get_settings
from app.utils.logger import configure_logger, logger

configure_logger()
settings = get_settings()

# ---------------------------------------------------------------------------
# App factory (with lifespan)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events: preload heavy models and connections on startup.
    """
    logger.info("Initializing heavy resources (lifespan)...")
    
    # Run preloading in separate threads/tasks to speed up boot
    loop = asyncio.get_event_loop()
    
    try:
        await asyncio.gather(
            loop.run_in_executor(None, embedder.preload_models),
            loop.run_in_executor(None, reranker.preload_models),
            loop.run_in_executor(None, get_client),
        )
        logger.success("All heavy resources initialized successfully")
    except Exception as exc:
        logger.error(f"Failed to initialize resources during startup: {exc}")
    
    yield
    # Cleanup (if any) goes here
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/css", StaticFiles(directory="app/static/css"), name="css")
app.mount("/js", StaticFiles(directory="app/static/js"), name="js")
app.mount("/images", StaticFiles(directory="app/static/images"), name="images")


@app.get("/", tags=["UI"])
async def read_root():
    """Serve the premium chatbot UI."""
    return FileResponse("app/static/index.html")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(5, ge=1, le=20)
    filters: dict[str, Any] | None = Field(None, description="Qdrant payload filters")


class SearchResult(BaseModel):
    text: str
    source_file: str
    chunk_type: str
    crop_category: str
    score: float | None = None
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(None, description="Omit to auto-generate a new session")
    filters: dict[str, Any] | None = None
    stream: bool = Field(True, description="Use SSE streaming if True")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    """Liveness check."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.mistral_model,
        "qdrant_collection": settings.qdrant_collection,
    }


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(req: SearchRequest) -> SearchResponse:
    """
    Raw hybrid retrieval — no LLM generation.
    Returns the top reranked chunks for the query.
    Useful for debugging retrieval quality.
    """
    try:
        chunks = retrieve_context(req.query, filters=req.filters)
    except Exception as exc:
        logger.error(f"/search error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    results = [
        SearchResult(
            text=c.get("text", ""),
            source_file=c.get("source_file", ""),
            chunk_type=c.get("chunk_type", ""),
            crop_category=c.get("crop_category", ""),
            score=c.get("rrf_score") or c.get("score"),
            rerank_score=c.get("rerank_score"),
        )
        for c in chunks[:req.top_k]
    ]

    return SearchResponse(query=req.query, results=results, total=len(results))


@app.post("/chat", tags=["Chat"])
async def chat(req: ChatRequest) -> Any:
    """
    Full RAG + ReAct agent chat.

    - If stream=True (default): returns SSE stream of text tokens.
    - If stream=False: returns JSON {session_id, answer, sources}.
    """
    session_id = req.session_id or str(uuid.uuid4())

    if req.stream:
        async def event_generator() -> AsyncGenerator[dict, None]:
            # First event: session id
            yield {"event": "session", "data": session_id}

            async for token in stream_agent(req.question, session_id, req.filters):
                yield {"event": "token", "data": json.dumps(token)}

            yield {"event": "done", "data": "[DONE]"}

        return EventSourceResponse(event_generator())

    # Non-streaming path
    try:
        answer = await run_agent(req.question, session_id, req.filters)
    except Exception as exc:
        logger.error(f"/chat error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    # Extract source files from context (run retrieval again for sources)
    try:
        chunks = retrieve_context(req.question, filters=req.filters)
        sources = list({c.get("source_file", "") for c in chunks if c.get("source_file")})
    except Exception:
        sources = []

    return ChatResponse(session_id=session_id, answer=answer, sources=sources)


@app.delete("/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str) -> dict[str, str]:
    """Clear conversation history for a session."""
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )
