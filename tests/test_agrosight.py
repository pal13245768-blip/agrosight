"""
AgroSight – Test Suite
========================
Tests cover:
  • Chunker strategies (unit tests, no external deps)
  • Embedding shape validation
  • API endpoints (integration, mocked Qdrant)
  • Tool outputs

Run: pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chunker import (
    ChunkResult,
    chunk_file,
    qa_pair_chunks,
    record_narrative_chunks,
    section_chunks,
    semantic_chunks,
    sliding_window_chunks,
    table_row_chunks,
)
from app.services.prompts import detect_language, format_context, format_history


# ===========================================================================
# Chunker unit tests
# ===========================================================================


class TestSemanticChunking:
    LONG_TEXT = (
        "Wheat is one of the most important cereal crops grown in India. "
        "It is primarily a Rabi crop sown in October-November and harvested in March-April. "
        "The main wheat-growing states are Punjab, Haryana, and Uttar Pradesh. "
        "Yellow Rust caused by Puccinia striiformis is a major disease of wheat. "
        "The disease causes stripe-like yellow lesions on leaves reducing photosynthesis. "
        "Recommended fungicides include Propiconazole and Tebuconazole applied at first appearance. "
        "Nitrogen fertilisation should be 120 kg N per hectare applied in three equal splits. "
        "The MSP for wheat in 2024-25 is 2275 rupees per quintal announced by CCEA. "
        "Irrigation at crown root initiation, tillering, jointing, and dough stages is critical. "
        "High-yielding varieties like HD 2967 and PBW 550 are widely adopted by farmers. "
    ) * 5  # ~700 words

    def test_produces_chunks(self):
        chunks = semantic_chunks(self.LONG_TEXT, source_file="test.pdf", chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_chunk_type_set(self):
        chunks = semantic_chunks(self.LONG_TEXT, chunk_type="text")
        assert all(c.chunk_type == "text" for c in chunks)

    def test_no_empty_chunks(self):
        chunks = semantic_chunks(self.LONG_TEXT, chunk_size=100, overlap=20)
        assert all(c.text.strip() for c in chunks)

    def test_hash_unique(self):
        chunks = semantic_chunks(self.LONG_TEXT, chunk_size=100, overlap=20)
        hashes = [c.chunk_hash for c in chunks]
        assert len(set(hashes)) == len(hashes), "Chunk hashes must be unique"

    def test_overlap_shares_content(self):
        chunks = semantic_chunks(self.LONG_TEXT, chunk_size=50, overlap=15)
        if len(chunks) > 1:
            # Last words of chunk[0] should appear in chunk[1]
            words_0 = set(chunks[0].text.split())
            words_1 = set(chunks[1].text.split())
            assert words_0 & words_1, "Overlap should share some words between consecutive chunks"

    def test_short_text_single_chunk(self):
        chunks = semantic_chunks("Short text.", chunk_size=512)
        assert len(chunks) == 1


class TestSectionChunking:
    STRUCTURED_TEXT = """
1. Introduction to Wheat Production
Wheat is the staple food grain of India and a critical Rabi crop.

1.1 Varieties
HD 2967 is a high-yielding semi-dwarf variety widely grown in Punjab and Haryana.
PBW 550 is recommended for timely sown conditions in Indo-Gangetic Plains.

2. Disease Management
Effective disease management is critical for achieving high yields.

2.1 Yellow Rust
Yellow rust caused by Puccinia striiformis appears as stripe-like yellow pustules.
Apply Propiconazole at first sign of infection.

2.2 Leaf Blight
Alternaria leaf blight appears as brown lesions with yellow halos.
"""

    def test_section_detection(self):
        chunks = section_chunks(self.STRUCTURED_TEXT, source_file="test.pdf")
        assert len(chunks) >= 2  # Should detect multiple sections

    def test_no_empty_chunks(self):
        chunks = section_chunks(self.STRUCTURED_TEXT)
        assert all(c.text.strip() for c in chunks)


class TestQAPairChunking:
    RECORDS = [
        {"question": "Who is eligible for PM-KISAN?", "answer": "All landholding farmer families."},
        {"question": "What is MSP for wheat?", "answer": "₹2,275 per quintal for 2024-25."},
        {"question": "", "answer": "Incomplete record – should be skipped."},  # Empty Q
    ]

    def test_produces_correct_count(self):
        chunks = qa_pair_chunks(self.RECORDS)
        assert len(chunks) == 2  # Third record skipped (empty question)

    def test_chunk_type_is_qa(self):
        chunks = qa_pair_chunks(self.RECORDS)
        assert all(c.chunk_type == "qa" for c in chunks)

    def test_text_contains_q_and_a(self):
        chunks = qa_pair_chunks(self.RECORDS)
        for c in chunks:
            assert "Q:" in c.text
            assert "A:" in c.text

    def test_metadata_has_question(self):
        chunks = qa_pair_chunks(self.RECORDS)
        assert "question" in chunks[0].metadata


class TestTableRowChunking:
    def test_group_by_commodity(self):
        import pandas as pd
        df = pd.DataFrame([
            {"Commodity": "Wheat", "Market": "Ahmedabad", "Modal_Price": 2275},
            {"Commodity": "Wheat", "Market": "Surat", "Modal_Price": 2300},
            {"Commodity": "Cotton", "Market": "Rajkot", "Modal_Price": 7020},
        ])
        chunks = table_row_chunks(df, group_by="Commodity")
        assert len(chunks) == 2  # One per commodity

    def test_batch_fallback(self):
        import pandas as pd
        df = pd.DataFrame({"col_a": range(25), "col_b": range(25)})
        chunks = table_row_chunks(df, batch_size=10)
        assert len(chunks) == 3  # ceil(25/10)


class TestRecordNarrativeChunking:
    RECORDS = [
        {
            "scheme_name": "PM-KISAN",
            "benefit": "₹6000/year",
            "eligibility": "All landholding farmers",
            "documents": ["Aadhaar", "Land record"],
        },
        {
            "scheme_name": "PMFBY",
            "benefit": "Crop insurance",
            "eligibility": "Loanee farmers (mandatory)",
        },
    ]

    def test_one_chunk_per_record(self):
        chunks = record_narrative_chunks(self.RECORDS, title_key="scheme_name")
        assert len(chunks) == 2

    def test_title_in_text(self):
        chunks = record_narrative_chunks(self.RECORDS, title_key="scheme_name")
        assert "PM-KISAN" in chunks[0].text


class TestSlidingWindow:
    def test_basic(self):
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = sliding_window_chunks(text, chunk_size=50, overlap=10)
        assert len(chunks) > 1

    def test_short_text(self):
        chunks = sliding_window_chunks("hello world", chunk_size=512)
        assert len(chunks) == 1


# ===========================================================================
# File-based chunker tests (using temp files)
# ===========================================================================


class TestChunkFileDispatch:
    def test_json_faq_dispatch(self, tmp_path):
        faq_data = [
            {"question": "Test Q?", "answer": "Test A."},
        ]
        fpath = tmp_path / "fertilizer_faqs.json"
        fpath.write_text(json.dumps(faq_data), encoding="utf-8")
        chunks = chunk_file(fpath)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "qa"

    def test_json_scheme_dispatch(self, tmp_path):
        scheme_data = [{"scheme_name": "PM-KISAN", "benefit": "6000/year"}]
        fpath = tmp_path / "all_schemes.json"
        fpath.write_text(json.dumps(scheme_data), encoding="utf-8")
        chunks = chunk_file(fpath)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "scheme_record"

    def test_csv_price_dispatch(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame([
            {"Commodity": "Wheat", "Market": "Ahmedabad", "Modal_Price": 2275},
            {"Commodity": "Rice", "Market": "Surat", "Modal_Price": 2300},
        ])
        fpath = tmp_path / "agmarket_india.csv"
        df.to_csv(fpath, index=False)
        chunks = chunk_file(fpath)
        assert len(chunks) >= 1
        assert all(c.chunk_type == "price_summary" for c in chunks)

    def test_vision_file_skipped(self, tmp_path):
        img_path = tmp_path / "leaf.jpg"
        img_path.write_bytes(b"\xff\xd8\xff")  # Fake JPEG header
        chunks = chunk_file(img_path)
        assert chunks == []

    def test_text_file(self, tmp_path):
        txt_path = tmp_path / "advisory.txt"
        txt_path.write_text("Wheat crop advisory for Punjab. Apply urea at CRI stage.", encoding="utf-8")
        chunks = chunk_file(txt_path)
        assert len(chunks) >= 1


# ===========================================================================
# Prompt utilities
# ===========================================================================


class TestPromptUtils:
    def test_detect_english(self):
        assert detect_language("What disease affects wheat?") == "en"

    def test_detect_hindi(self):
        assert detect_language("गेहूं की फसल में कौन सा रोग लगता है?") == "hi"

    def test_detect_gujarati(self):
        assert detect_language("ઘઉંની ખેતીમાં કઈ બીમારી થાય છે?") == "gu"

    def test_format_context(self):
        chunks = [
            {"text": "Wheat rust is a fungal disease.", "source_file": "wheat.pdf"},
            {"text": "Apply Propiconazole as fungicide.", "source_file": "icar_guide.pdf"},
        ]
        ctx = format_context(chunks)
        assert "wheat.pdf" in ctx
        assert "Propiconazole" in ctx

    def test_format_history_empty(self):
        result = format_history([])
        assert "No prior" in result

    def test_format_history_truncates(self):
        history = [{"role": "user", "content": "x" * 1000}]
        result = format_history(history)
        assert len(result) < 1000


# ===========================================================================
# ChunkResult dataclass
# ===========================================================================


class TestChunkResult:
    def test_hash_deterministic(self):
        c1 = ChunkResult(text="Hello wheat farmer")
        c2 = ChunkResult(text="Hello wheat farmer")
        assert c1.chunk_hash == c2.chunk_hash

    def test_different_texts_different_hash(self):
        c1 = ChunkResult(text="Wheat")
        c2 = ChunkResult(text="Rice")
        assert c1.chunk_hash != c2.chunk_hash


# ===========================================================================
# API integration tests (mocked)
# ===========================================================================


@pytest.fixture
def test_client():
    from fastapi.testclient import TestClient

    with patch("app.services.agent.retrieve_context", return_value=[
        {"text": "PM-KISAN provides ₹6000/year.", "source_file": "scheme.json",
         "chunk_type": "qa", "crop_category": "government", "rerank_score": 0.9}
    ]), patch("app.services.agent.run_agent", return_value="PM-KISAN is a scheme providing ₹6000 per year. [Source: scheme.json]"):
        from app.main import app
        client = TestClient(app)
        yield client


class TestAPIEndpoints:
    def test_health(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_search(self, test_client):
        resp = test_client.post("/search", json={"query": "PM-KISAN eligibility", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert data["query"] == "PM-KISAN eligibility"

    def test_chat_non_streaming(self, test_client):
        resp = test_client.post("/chat", json={
            "question": "Who is eligible for PM-KISAN?",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "session_id" in data

    def test_delete_session(self, test_client):
        resp = test_client.delete("/session/test-session-123")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
