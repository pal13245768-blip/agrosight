"""
AgroSight – LangGraph ReAct Agent
===================================
Orchestrates the full RAG pipeline:
  1. Encode query (bge-m3)
  2. Hybrid retrieve from Qdrant
  3. Rerank with cross-encoder
  4. Build prompt with context + history
  5. Run ReAct agent (Ollama LLM + 3 tools)
  6. Stream response tokens

The agent has three tools:
  • weather   → get_weather_advisory
  • price     → get_mandi_price
  • fertilise → fertiliser_calculator
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, AIMessageChunk, ChatMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
import langchain_mistralai.chat_models as mistral_chat_models
from langgraph.prebuilt import create_react_agent

from app.services.agro_tools import (
    fertiliser_calculator,
    get_mandi_price,
    get_weather_advisory,
)
from app.services.embedder import encode_query, encode_sparse
from app.services.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    detect_language,
    get_language_name,
    format_context,
    format_history,
)
from app.services.reranker import rerank
from app.services.session_store import append_turn, get_history
from app.services.vector_store import hybrid_search
from app.utils.config import get_settings
from app.utils.logger import logger

settings = get_settings()


# Fix a Mistral tool-calling conversion bug in langchain_mistralai.
# The package duplicates tool calls when converting AIMessages to Mistral format,
# which causes `Duplicate tool call id in assistant message` errors.
def _patched_convert_message_to_mistral_chat_message(message):
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        message_dict: dict[str, Any] = {"role": "assistant"}
        tool_calls: list[dict[str, Any]] = []

        if message.tool_calls or message.invalid_tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(
                    mistral_chat_models._format_tool_call_for_mistral(tool_call)
                )
            for invalid_tool_call in message.invalid_tool_calls:
                tool_calls.append(
                    mistral_chat_models._format_invalid_tool_call_for_mistral(invalid_tool_call)
                )
        elif "tool_calls" in message.additional_kwargs:
            for tc in message.additional_kwargs["tool_calls"]:
                chunk = {
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                }
                if _id := tc.get("id"):
                    chunk["id"] = _id
                tool_calls.append(chunk)

        if tool_calls:
            message_dict["tool_calls"] = tool_calls
        if tool_calls and message.content:
            message_dict["content"] = ""
        else:
            message_dict["content"] = message.content
        if "prefix" in message.additional_kwargs:
            message_dict["prefix"] = message.additional_kwargs["prefix"]
        return message_dict
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "name": message.name,
            "tool_call_id": mistral_chat_models._convert_tool_call_id_to_mistral_compatible(
                message.tool_call_id
            ),
        }
    raise ValueError(f"Got unknown type {message}")

mistral_chat_models._convert_message_to_mistral_chat_message = (
    _patched_convert_message_to_mistral_chat_message
)


# ---------------------------------------------------------------------------
# LangChain-compatible tool wrappers
# ---------------------------------------------------------------------------


@tool
async def weather_tool(location: str) -> str:
    """Get current weather and agronomy advisory for a location."""
    result = await get_weather_advisory(location)
    return str(result)


@tool
async def mandi_price_tool(commodity: str, market: str = "", state: str = "Gujarat") -> str:
    """Get today's mandi price for a commodity."""
    result = await get_mandi_price(commodity, market, state)
    return str(result)


@tool
async def fertiliser_tool(crop: str, area_acres: float, fertiliser: str = "urea", nutrient: str = "N") -> str:
    """Calculate fertiliser dose for a crop on given area."""
    result = await fertiliser_calculator(crop, area_acres, fertiliser, nutrient)
    return str(result)


TOOLS = [weather_tool, mandi_price_tool, fertiliser_tool]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def _get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        api_key=settings.mistral_api_key,
        model=settings.mistral_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


# ---------------------------------------------------------------------------
# Core retrieval pipeline
# ---------------------------------------------------------------------------


def retrieve_context(query: str, filters: dict | None = None) -> list[dict[str, Any]]:
    """
    Full retrieval pipeline:
      encode → hybrid search → rerank → return top-5 chunks
    """
    query_vec = encode_query(query)
    sparse_weights = encode_sparse([query])[0] if "bge-m3" in settings.embedding_model.lower() else {}

    candidates = hybrid_search(
        query_vector=query_vec,
        sparse_weights=sparse_weights or None,
        top_k=settings.retrieval_top_k,
        filters=filters,
    )

    reranked = rerank(query, candidates, top_k=5)
    logger.debug(f"Retrieved {len(candidates)} candidates → reranked to {len(reranked)}")
    return reranked


# ---------------------------------------------------------------------------
# Agent run (non-streaming)
# ---------------------------------------------------------------------------


async def run_agent(question: str, session_id: str = "default", filters: dict | None = None) -> str:
    """Synchronous execution of the agent for a single turn."""
    res = await run_agent_with_metadata(question, session_id, filters)
    return res["answer"]


async def run_agent_with_metadata(
    question: str, session_id: str = "default", filters: dict | None = None
) -> dict[str, Any]:
    """
    Execution of the agent with detailed metadata capture.
    Returns: {"answer": str, "tool_calls": list[str], "chunks": list[dict]}
    """
    # Retrieve context
    chunks = retrieve_context(question, filters=filters)
    context_str = format_context(chunks)

    # Load session history
    history = get_history(session_id)
    history_str = format_history(history)

    # Detect language and lock the prompt
    lang_code = detect_language(question)
    lang_name = get_language_name(lang_code)

    # Build user message
    user_msg = RAG_USER_TEMPLATE.format(
        context=context_str,
        history=history_str,
        question=question,
        language=lang_name,
    )

    # 5. Build LangGraph ReAct agent
    llm = _get_llm()
    agent = create_react_agent(llm, TOOLS)

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    # 6. Run agent (asynchronous)
    result = await agent.ainvoke(
        {"messages": messages},
        config={"recursion_limit": settings.max_agent_iterations * 2},
    )

    # Extract final text answer and tool calls
    answer = ""
    tool_names = []
    
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            if msg.content:
                answer = str(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_names.append(tc["name"])

    #  Persist to session store
    append_turn(session_id, "user", question)
    append_turn(session_id, "assistant", answer)

    return {
        "answer": answer,
        "tool_calls": tool_names,
        "chunks": chunks,
    }


# ---------------------------------------------------------------------------
# Streaming agent run (SSE-compatible)
# ---------------------------------------------------------------------------


async def stream_agent(
    question: str,
    session_id: str,
    filters: dict | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields text tokens for SSE streaming.
    Runs retrieval synchronously then streams the LLM response.
    """
    # Run retrieval in thread pool (CPU-bound embedding)
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, retrieve_context, question, filters)
    context_str = format_context(chunks)

    history = get_history(session_id)
    history_str = format_history(history)

    # Detect language and lock the prompt
    lang_code = detect_language(question)
    lang_name = get_language_name(lang_code)
    logger.debug(f"Detected language: {lang_code} ({lang_name})")

    user_msg = RAG_USER_TEMPLATE.format(
        context=context_str,
        history=history_str,
        question=question,
        language=lang_name,
    )

    llm = _get_llm()
    agent = create_react_agent(llm, TOOLS)

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    full_answer_parts: list[str] = []

    # Stream from the agent with buffering for smoother delivery
    buffer = ""
    async for msg, metadata in agent.astream(
        {"messages": messages},
        stream_mode="messages",
        config={"recursion_limit": settings.max_agent_iterations * 2},
    ):
        if isinstance(msg, AIMessageChunk) and msg.content:
            token = str(msg.content)
            full_answer_parts.append(token)
            buffer += token

            # Aggressive buffering for premium streaming experience
            # Yield only on paragraph breaks or when the buffer is sufficiently long
            if "\n" in token:
                yield buffer
                buffer = ""
            elif len(buffer) > 40:
                yield buffer
                buffer = ""

    # Yield any remaining content in buffer
    if buffer:
        yield buffer

    # Persist after streaming completes
    full_answer = "".join(full_answer_parts)
    append_turn(session_id, "user", question)
    append_turn(session_id, "assistant", full_answer)
