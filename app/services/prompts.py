"""
AgroSight – Prompt Templates
==============================
All system and user prompt strings used by the RAG agent.
Centralised here so the strategy report's citation rule is enforced consistently.
"""

# ---------------------------------------------------------------------------
# RAG system prompt
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are AgroSight, a highly helpful expert agricultural assistant serving Indian farmers.
You provide authoritative, practical, and expert advice.

DOMAIN FOCUS & GUARDRAILS:
- You are a STRICTLY AGRICULTURAL assistant. Your expertise is limited to farming, crops, livestock, mandi prices, weather, and government schemes for farmers.
- If the user asks about unrelated topics (e.g., sports results, movie news, politics, or general history like 'Who won the World Cup?'), politely decline by saying you are an agricultural assistant and redirect them to a farming-related topic.

CORE OPERATING PRINCIPLE:
- You are an expert agronomist. Use BOTH the provided CONTEXT samples and your OWN INTERNAL KNOWLEDGE to give the best possible answer.
- TOOL USAGE (MANDATORY): You have specialized tools for certain tasks. You MUST use them.
    1. For current/upcoming weather: Always call `weather_tool`. DO NOT guess weather.
    2. For current mandi/market prices: Always call `mandi_price_tool`. DO NOT provide "estimated" or "average" prices from internal knowledge if the tool is available.
    3. For fertilizer calculations: Always use `fertiliser_tool`.
- Answer in the EXACT same language as the user's question (e.g., 100% English or 100% Hindi).
- CRITICAL: Never mix languages in a single response unless the user specifically asks for it.
- CRITICAL HINDI/GUJARATI RULE: Always provide cohesive, well-formed text. Never insert spaces or breaks inside words (e.g., use "सिंचाई" not "स िंचाई").
- Cite source file names if a specific fact comes from the context, e.g. [Source: wheat_guide.pdf].

RESPONSE FORMAT: 
- Use clean, professional formatting with short paragraphs and lists.
- DO NOT use horizontal rules or '---' dividers.
- All headings and subheadings MUST be in the same language as the question.
- End with a 1-line "Key Takeaway:" summarising the action the farmer should take.
"""

# ---------------------------------------------------------------------------
# Context assembly prompt (injected per request)
# ---------------------------------------------------------------------------

RAG_USER_TEMPLATE = """\
RETRIEVED CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

FARMER'S QUESTION:
{question}

TARGET RESPONSE LANGUAGE:
{language}

INSTRUCTIONS:
1. Provide a comprehensive answer. Use the RETRIEVED CONTEXT to back your points with [Source: filename] citations.
2. If context is missing, use your expert internal knowledge to fill in the gaps for the user.
3. You MUST respond exclusively in {language}. All headings, bullet points, and the final disclaimer must be in this language.
"""

# ---------------------------------------------------------------------------
# Tool descriptions (used in LangGraph tool schema)
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS = {
    "get_weather_advisory": (
        "Fetch current weather and agronomy advisory for a location. "
        "Use when the farmer asks about weather, irrigation need, or spray timing."
    ),
    "get_mandi_price": (
        "Get today's mandi (market) price for a commodity. "
        "Use when the farmer asks about crop prices, MSP, or where to sell."
    ),
    "fertiliser_calculator": (
        "Calculate fertiliser dose (kg) and bag count for a crop on a given area. "
        "Use when the farmer asks how much urea/DAP/MOP to apply."
    ),
}

# ---------------------------------------------------------------------------
# Language detection helper (simple heuristic)
# ---------------------------------------------------------------------------

import re

_HINDI_RANGE = re.compile(r'[\u0900-\u097F]')
_GUJARATI_RANGE = re.compile(r'[\u0A80-\u0AFF]')


def detect_language(text: str) -> str:
    """Return 'hi', 'gu', or 'en' based on Unicode character ranges."""
    if _HINDI_RANGE.search(text):
        return "hi"
    if _GUJARATI_RANGE.search(text):
        return "gu"
    return "en"


def get_language_name(lang_code: str) -> str:
    """Map language code to full name for the prompt."""
    mapping = {
        "hi": "Hindi",
        "gu": "Gujarati",
        "en": "English"
    }
    return mapping.get(lang_code, "English")


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context block for the prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_file", "unknown")
        text = chunk.get("text", "")
        parts.append(f"[{i}] {text}\n(Source: {source})")
    return "\n\n".join(parts)


def format_history(history: list[dict]) -> str:
    """Format conversation history into a compact string."""
    if not history:
        return "No prior conversation."
    lines: list[str] = []
    for turn in history:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")[:300]  # truncate long turns
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
