import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.agent import run_agent_with_metadata
from app.utils.config import get_settings
from app.utils.logger import configure_logger, logger
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

configure_logger()
settings = get_settings()

# ---------------------------------------------------------------------------
# Test Matrix (Queries, Categories, Expected Tools)
# ---------------------------------------------------------------------------

TEST_MATRIX = [
    {
        "id": "RAG-EN-01",
        "category": "RAG",
        "language": "English",
        "query": "What are the common pests in high growth Mango trees?",
        "expected_tool": None,
        "ground_truth_keywords": ["pest", "mango", "anthracnose", "hopper", "mealybug"],
    },
    {
        "id": "TOOL-WEATH-01",
        "category": "Tool",
        "language": "English",
        "query": "What is the weather like in Ahmedabad and give me agricultural advice?",
        "expected_tool": "weather_tool",
        "ground_truth_keywords": ["weather", "ahmedabad", "advisory", "temperature"],
    },
    {
        "id": "TOOL-MANDI-01",
        "category": "Tool",
        "language": "English",
        "query": "Give me the latest mandi price for Wheat in Gujarat.",
        "expected_tool": "mandi_price_tool",
        "ground_truth_keywords": ["price", "wheat", "gujarat", "mandi", "Rs."],
    },
    {
        "id": "TOOL-FERT-01",
        "category": "Tool",
        "language": "English",
        "query": "How much urea do I need for 5 acres of wheat?",
        "expected_tool": "fertiliser_tool",
        "ground_truth_keywords": ["urea", "kg", "acres", "bags"],
    },
    {
        "id": "RAG-HI-01",
        "category": "RAG",
        "language": "Hindi",
        "query": "गेहूं की खेती के मुख्य चरण क्या हैं?",
        "expected_tool": None,
        "ground_truth_keywords": ["फसल", "तैयारी", "बुवाई", "सिंचाई"],
    },
    {
        "id": "RAG-GU-01",
        "category": "RAG",
        "language": "Gujarati",
        "query": "ઘઉંની ખેતીમાં કયા કયા ખાતરો વાપરવા?",
        "expected_tool": None,
        "ground_truth_keywords": ["ખાતર", "યુરિયા", "ડીએપી"],
    },
    {
        "id": "GUARD-OFFTOPIC-01",
        "category": "Guardrail",
        "language": "English",
        "query": "Who won the World Cup 2024?",
        "expected_tool": None,
        "ground_truth_keywords": ["agricultural", "assistant", "cannot answer", "farming"],
    },
    {
        "id": "GUARD-OFFTOPIC-02",
        "category": "Guardrail",
        "language": "English",
        "query": "Who is the most famous Bollywood actor?",
        "expected_tool": None,
        "ground_truth_keywords": ["agricultural", "assistant", "cannot answer", "farming"],
    },
    {
        "id": "FALLBACK-01",
        "category": "Fallback",
        "language": "English",
        "query": "How to grow hydroponic lettuce in India?",
        "expected_tool": None,
        "ground_truth_keywords": ["hydroponic", "lettuce", "nutrient", "water"],
    },
]

# ---------------------------------------------------------------------------
# Evaluator (Mistral as Judge)
# ---------------------------------------------------------------------------

EVAL_PROMPT = """\
You are an expert impartial auditor evaluating an Agricultural AI Assistant named AgroSight.
You will be given a user QUESTION, the AI's ANSWER, and the retrieved CONTEXT.

Score the AI Answer on a scale of 1 to 5 (Integer) based on:
1. ACCURACY: Is the information technically sound?
2. RELEVANCE: Does it answer the user's question directly?
3. LANGUAGE: Does it match the question's language (English/Hindi/Gujarati)?
4. TONE: Is it professional, authoritative, and helpful?

Output your evaluation in JSON format:
{{
    "accuracy_score": int,
    "relevance_score": int,
    "language_score": int,
    "tone_score": int,
    "overall_score": float,
    "feedback": "string"
}}
"""

async def evaluate_answer(question: str, answer: str, context: str) -> dict:
    llm = ChatMistralAI(api_key=settings.mistral_api_key, model=settings.mistral_model, temperature=0)
    user_str = f"QUESTION: {question}\n\nCONTEXT: {context}\n\nANSWER: {answer}"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke([
                SystemMessage(content=EVAL_PROMPT),
                HumanMessage(content=user_str)
            ])
            # Find JSON in response
            raw_text = response.content
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            return json.loads(raw_text[start:end])
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                logger.warning(f"Rate limit hit during eval. Waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            logger.error(f"Evaluation failed: {e}")
            return {"overall_score": 0.0, "feedback": f"Error: {e}"}


# ---------------------------------------------------------------------------
# Audit Runner
# ---------------------------------------------------------------------------

async def run_audit():
    logger.info("Starting AgroSight System Audit...")
    report_rows = []
    
    for case in TEST_MATRIX:
        qid = case["id"]
        query = case["query"]
        logger.info(f"Running Case {qid}: {query[:50]}...")
        
        # Avoid rate limits (be generous)
        await asyncio.sleep(8)
        
        res = None
        max_agent_retries = 3
        for attempt in range(max_agent_retries):
            try:
                res = await run_agent_with_metadata(query, session_id=f"audit_{qid}")
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_agent_retries - 1:
                    wait = (attempt + 1) * 15
                    logger.warning(f"Rate limit hit during AGENT RUN. Waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"Agent failed for case {qid} after {attempt+1} attempts: {e}")
                break
        
        if not res:
            report_rows.append({
                "id": qid,
                "category": case["category"],
                "lang": case["language"],
                "query": query,
                "tool": "❌ ERROR (Agent Failed)",
                "recall": "0%",
                "score": 0,
                "feedback": "Agent call failed continuously."
            })
            continue

        answer = res["answer"]
        tools_called = res["tool_calls"]
        chunks = res["chunks"]
        
        # 1. Retrieval Quality (Keyword Recall)
        content_found = " ".join([c.get("text", "") for c in chunks]).lower()
        hits = sum(1 for kw in case["ground_truth_keywords"] if kw.lower() in (content_found + answer.lower()))
        recall_pct = (hits / len(case["ground_truth_keywords"])) * 100
        
        # 2. Tool Calling
        tool_status = "✅ Correct"
        if case["expected_tool"]:
            if case["expected_tool"] in tools_called:
                tool_status = f"✅ {case['expected_tool']}"
            else:
                tool_status = f"❌ Missing {case['expected_tool']}"
        elif tools_called:
            tool_status = f"⚠️ Unexpected: {', '.join(tools_called)}"
        else:
            tool_status = "✅ None (Correct)"

        # 3. Qualitative Scoring (LLM Judge)
        context_str = "\n".join([f"- {c.get('text', '')}" for c in chunks[:3]])
        eval_res = await evaluate_answer(query, answer, context_str)
        
        report_rows.append({
            "id": qid,
            "category": case["category"],
            "lang": case["language"],
            "query": query,
            "tool": tool_status,
            "recall": f"{recall_pct:.0f}%",
            "score": eval_res.get("overall_score", 0),
            "feedback": eval_res.get("feedback", "")[:100] + "..."
        })
        
    # Generate Markdown Table
    md = "# AgroSight System Audit Matrix\n\n"
    md += "| ID | Category | Language | Tool Accuracy | Recall | Quality Score | Feedback |\n"
    md += "|----|----------|----------|---------------|--------|---------------|----------|\n"
    for row in report_rows:
        md += f"| {row['id']} | {row['category']} | {row['lang']} | {row['tool']} | {row['recall']} | {row['score']}/5 | {row['feedback']} |\n"
    
    Path("audit_matrix.md").write_text(md, encoding="utf-8")
    logger.success(f"Audit complete! Matrix saved to audit_matrix.md")

if __name__ == "__main__":
    asyncio.run(run_audit())
