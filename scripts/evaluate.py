"""
AgroSight – RAGAS Evaluation Suite
=====================================
Evaluates retrieval and generation quality using the RAGAS framework.
Measures: Faithfulness, Answer Relevancy, Context Recall, Context Precision.

Usage:
    python -m scripts.evaluate [--eval-set data/eval_set.json] [--output eval_results.json]

Eval set JSON format:
    [
        {
            "question": "What disease causes yellow stripes on wheat leaves?",
            "ground_truth": "Wheat Yellow Rust (Puccinia striiformis) causes yellow stripe patterns.",
            "language": "en"
        },
        ...
    ]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.agent import retrieve_context, run_agent
from app.services.prompts import format_context
from app.utils.config import get_settings
from app.utils.logger import configure_logger, logger

configure_logger()
settings = get_settings()

# ---------------------------------------------------------------------------
# Built-in evaluation set (representative AgroSight queries)
# ---------------------------------------------------------------------------

BUILT_IN_EVAL_SET = [
    # English – crop disease
    {
        "question": "What disease causes yellow stripe patterns on wheat leaves?",
        "ground_truth": "Yellow Stripe Rust caused by Puccinia striiformis causes yellow to orange pustules in stripes on wheat leaves.",
        "language": "en",
        "category": "disease",
    },
    # English – government scheme
    {
        "question": "Who is eligible for PM-KISAN?",
        "ground_truth": "All landholding farmer families with cultivable land are eligible for PM-KISAN, providing ₹6,000 per year.",
        "language": "en",
        "category": "government",
    },
    # English – fertiliser
    {
        "question": "How much urea should I apply to wheat per hectare?",
        "ground_truth": "Approximately 260 kg urea per hectare should be applied for wheat in split doses corresponding to 120 kg N/ha requirement.",
        "language": "en",
        "category": "fertiliser",
    },
    # English – soil
    {
        "question": "What crops grow best in black cotton soil?",
        "ground_truth": "Black cotton soil (Regur) is best suited for cotton, soybean, wheat, jowar, and chickpea.",
        "language": "en",
        "category": "soil",
    },
    # English – price
    {
        "question": "What is the MSP for wheat?",
        "ground_truth": "The MSP for wheat for 2024-25 is ₹2,275 per quintal as announced by CCEA.",
        "language": "en",
        "category": "price",
    },
    # Hindi – disease
    {
        "question": "गेहूं की पत्तियों पर पीली धारियां किस रोग के कारण होती हैं?",
        "ground_truth": "पीली धारियां गेहूं रस्ट (Puccinia striiformis) के कारण होती हैं।",
        "language": "hi",
        "category": "disease",
    },
    # Hindi – scheme
    {
        "question": "पीएम किसान योजना के लिए कौन पात्र है?",
        "ground_truth": "सभी भूमि धारक किसान परिवार पीएम किसान के लिए पात्र हैं। योजना वार्षिक ₹6000 प्रदान करती है।",
        "language": "hi",
        "category": "government",
    },
    # Gujarati – fertiliser
    {
        "question": "ઘઉં માટે યુરિયા કેટલો વાપરવો?",
        "ground_truth": "ઘઉં માટે હેક્ટર દીઠ 260 કિલો યુરિયા ત્રણ હપ્તામાં આપવો.",
        "language": "gu",
        "category": "fertiliser",
    },
    # English – insurance
    {
        "question": "What is the premium rate for PMFBY crop insurance for Kharif crops?",
        "ground_truth": "The farmer's premium for PMFBY is 2% of sum insured for Kharif crops, with government paying the remaining actuarial premium.",
        "language": "en",
        "category": "government",
    },
    # English – pest
    {
        "question": "How do I control cotton bollworm?",
        "ground_truth": "Control bollworm with Chlorantraniliprole 18.5 SC, use pheromone traps, and plant Bt cotton varieties.",
        "language": "en",
        "category": "pest",
    },
]


# ---------------------------------------------------------------------------
# Simple scoring (no external RAGAS dependency)
# ---------------------------------------------------------------------------


def _token_overlap(text_a: str, text_b: str) -> float:
    """Compute token-level F1 overlap between two strings."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    precision = len(intersection) / len(tokens_a)
    recall = len(intersection) / len(tokens_b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_answer(answer: str, ground_truth: str) -> dict[str, float]:
    """Compute simple local scores (token F1, answer length)."""
    f1 = _token_overlap(answer, ground_truth)
    return {
        "token_f1": round(f1, 4),
        "answer_length": len(answer.split()),
        "has_source_citation": "[Source:" in answer or "(Source:" in answer,
    }


def score_retrieval(chunks: list[dict], ground_truth: str) -> dict[str, float]:
    """Check if retrieved context contains ground truth keywords."""
    context_text = " ".join(c.get("text", "") for c in chunks).lower()
    gt_keywords = [w for w in ground_truth.lower().split() if len(w) > 4]
    if not gt_keywords:
        return {"context_recall": 0.0}
    hits = sum(1 for kw in gt_keywords if kw in context_text)
    recall = hits / len(gt_keywords)
    return {"context_recall": round(recall, 4)}


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate(
    eval_set: list[dict],
    session_prefix: str = "eval",
) -> dict[str, Any]:
    """
    Run evaluation over *eval_set*.
    Returns aggregated metrics and per-question results.
    """
    results: list[dict] = []
    start_total = time.perf_counter()

    for i, item in enumerate(eval_set):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        language = item.get("language", "en")
        category = item.get("category", "general")
        session_id = f"{session_prefix}_{i}"

        logger.info(f"[{i+1}/{len(eval_set)}] [{language}] {question[:60]}…")

        t0 = time.perf_counter()
        try:
            # Retrieval
            chunks = retrieve_context(question)
            retrieval_scores = score_retrieval(chunks, ground_truth)

            # Generation
            answer = run_agent(question, session_id)
            latency_ms = round((time.perf_counter() - t0) * 1000)

            answer_scores = score_answer(answer, ground_truth)

            result = {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "language": language,
                "category": category,
                "latency_ms": latency_ms,
                "num_chunks_retrieved": len(chunks),
                **retrieval_scores,
                **answer_scores,
                "status": "ok",
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000)
            result = {
                "question": question,
                "language": language,
                "category": category,
                "latency_ms": latency_ms,
                "error": str(exc),
                "status": "error",
            }
            logger.error(f"Eval error for q{i}: {exc}")

        results.append(result)
        logger.debug(f"  → latency={latency_ms}ms  f1={result.get('token_f1', 'N/A')}")

    total_time = round(time.perf_counter() - start_total, 2)

    # Aggregate
    ok_results = [r for r in results if r["status"] == "ok"]
    metrics = {
        "total_questions": len(eval_set),
        "successful": len(ok_results),
        "errors": len(results) - len(ok_results),
        "total_time_s": total_time,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results)) if results else 0,
        "avg_token_f1": round(sum(r.get("token_f1", 0) for r in ok_results) / len(ok_results), 4) if ok_results else 0,
        "avg_context_recall": round(sum(r.get("context_recall", 0) for r in ok_results) / len(ok_results), 4) if ok_results else 0,
        "citation_rate": round(sum(r.get("has_source_citation", 0) for r in ok_results) / len(ok_results), 4) if ok_results else 0,
    }

    # By language
    for lang in ("en", "hi", "gu"):
        lang_res = [r for r in ok_results if r.get("language") == lang]
        if lang_res:
            metrics[f"avg_token_f1_{lang}"] = round(
                sum(r.get("token_f1", 0) for r in lang_res) / len(lang_res), 4
            )

    # By category
    cats = set(r.get("category", "general") for r in ok_results)
    for cat in cats:
        cat_res = [r for r in ok_results if r.get("category") == cat]
        if cat_res:
            metrics[f"avg_f1_{cat}"] = round(
                sum(r.get("token_f1", 0) for r in cat_res) / len(cat_res), 4
            )

    return {"metrics": metrics, "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroSight RAGAS Evaluation")
    parser.add_argument("--eval-set", default=None, help="Path to eval set JSON file")
    parser.add_argument("--output", default="eval_results.json", help="Output file for results")
    args = parser.parse_args()

    if args.eval_set:
        eval_data = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    else:
        logger.info("No eval set provided – using built-in AgroSight eval set")
        eval_data = BUILT_IN_EVAL_SET

    report = evaluate(eval_data)

    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.success(f"\n{'='*60}")
    logger.success("EVALUATION SUMMARY")
    logger.success(f"{'='*60}")
    for k, v in report["metrics"].items():
        logger.success(f"  {k:40s}: {v}")
    logger.success(f"\nFull results saved to: {args.output}")
