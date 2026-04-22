# 🌾 AgroSight RAG System

**Production-grade agricultural Retrieval-Augmented Generation API** serving Indian farmers with crop advisory, disease diagnosis, mandi prices, scheme information, and fertiliser calculations — in English, Hindi, and Gujarati.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AgroSight RAG Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Data Sources                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │113 PDFs  │ │ Mandi CSV│ │ Scheme   │ │ Disease  │ │20K Images│  │
│  │(1 GB)    │ │ USDA/FAO │ │ FAQs JSON│ │ KB JSON  │ │(Vision)  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       │             │             │             │             │        │
│       ▼             ▼             ▼             ▼             ▼        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │           Hybrid Auto-Select Chunker (chunker.py)               │ │
│  │  Semantic | Section | Q&A | Table | Record | Sliding Window     │ │
│  └─────────────────────────┬───────────────────────────────────────┘ │
│                             │                                          │
│                             ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              BAAI/bge-m3 Embedder (1024d, multilingual)         │ │
│  │         Dense vectors + BM25 sparse weights (hybrid)            │ │
│  └─────────────────────────┬───────────────────────────────────────┘ │
│                             │                                          │
│                             ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │           Qdrant Cloud (agricultural_knowledge_v2)               │ │
│  │          Cosine + Hybrid Dense/Sparse Search                    │ │
│  └─────────────────────────┬───────────────────────────────────────┘ │
│                             │                                          │
│  ┌──────────────────────────▼──────────────────────────────────────┐ │
│  │           Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)       │ │
│  └──────────────────────────┬──────────────────────────────────────┘ │
│                             │                                          │
│  ┌──────────────────────────▼──────────────────────────────────────┐ │
│  │   LangGraph ReAct Agent (Ollama qwen3.5:9b)                     │ │
│  │   Tools: 🌦 Weather  💰 Mandi Price  🌱 Fertiliser Calculator   │ │
│  └──────────────────────────┬──────────────────────────────────────┘ │
│                             │                                          │
│  ┌──────────────────────────▼──────────────────────────────────────┐ │
│  │   FastAPI + SSE Streaming  │  Redis Session Store (5 turns)     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.11+
python --version

# Ollama (local LLM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:9b
ollama serve

# Redis (optional – falls back to in-memory)
docker run -d -p 6379:6379 redis:7.2-alpine
```

### 2. Install & Configure

```bash
git clone https://github.com/yourorg/agrosight.git
cd agrosight

pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
# Edit .env: add QDRANT_URL, QDRANT_API_KEY, OPENWEATHER_API_KEY, etc.
```

### 3. Seed Data & Ingest

```bash
# Step 1: Download/seed knowledge files (disease, schemes, soil, FAQs)
make download-knowledge

# Step 2: Download live data (optional – needs API keys)
make download-weather    # OpenWeatherMap
make download-mandi      # data.gov.in mandi prices

# Step 3: Place your PDF books in data/books/
# (113 agricultural PDFs as described in the strategy report)

# Step 4: Run full ingestion pipeline
make ingest
# → Chunks all files → Embeds with bge-m3 → Upserts to Qdrant
```

### 4. Start API

```bash
# Development (with hot-reload)
make dev-server

# Production (Docker)
make docker-up
```

API is live at **http://localhost:8000**  
Swagger docs at **http://localhost:8000/docs**

---

## API Endpoints

### `GET /health`
```json
{
  "status": "ok",
  "version": "1.0.0",
  "embedding_model": "BAAI/bge-m3",
  "llm_model": "qwen3.5:9b",
  "qdrant_collection": "agricultural_knowledge_v2"
}
```

### `POST /search` — Raw retrieval (no LLM)
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "wheat rust management", "top_k": 5}'
```

```json
{
  "query": "wheat rust management",
  "results": [
    {
      "text": "Yellow Rust caused by Puccinia striiformis...",
      "source_file": "data/raw/pest_disease/disease_knowledge.json",
      "chunk_type": "disease_record",
      "crop_category": "pest_disease",
      "rerank_score": 0.92
    }
  ],
  "total": 5
}
```

### `POST /chat` — SSE Streaming (default)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "गेहूं में पीला रतुआ कैसे रोकें?", "stream": true}'
```

Streams SSE events:
```
event: session
data: abc-123-session-id

event: token
data: गेहूं

event: token
data: में

... (more tokens)

event: done
data: [DONE]
```

### `POST /chat` — Non-streaming
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What subsidy is available for drip irrigation?", "stream": false}'
```

```json
{
  "session_id": "abc-123",
  "answer": "Under PMKSY-PDMC, drip irrigation is subsidised at 55% for small/marginal farmers... [Source: scheme_faqs.json]",
  "sources": ["data/raw/government/scheme_faqs.json"]
}
```

### `DELETE /session/{session_id}`
Clears conversation history for multi-turn sessions.

---

## Data Sources & Chunking Strategy

| Data Source | Files | Chunking Strategy | Qdrant Filter |
|---|---|---|---|
| `data/books/*.pdf` (113 books) | ~1 GB | Section → Semantic fallback (512w, 64 overlap) | `chunk_type=text` |
| `data/raw/government/scheme_faqs.json` | FAQs | Q&A pair (1 per pair) | `chunk_type=qa, crop_category=government` |
| `data/raw/government/all_schemes.json` | Schemes | Record narrative (1 per scheme) | `chunk_type=scheme_record` |
| `data/collected/agmarket_*.csv` | Prices | Table chunks per commodity | `chunk_type=price_summary` |
| `data/raw/soil/indian_soil_classification.json` | Soil | Section narrative | `chunk_type=soil_classification` |
| `data/raw/pest_disease/disease_knowledge.json` | Diseases | Record narrative | `chunk_type=disease_record` |
| `data/raw/weather/openweather/agro_advisories.json` | Weather | Advisory record | `chunk_type=advisory` |
| `data/raw/fertilizer/fertilizer_faqs.json` | Fertiliser | Q&A pair | `chunk_type=qa, crop_category=fertilizer` |
| `data/raw/pest_disease/*.jpg` | 20,643 images | **SKIP** – vision pipeline | image_index (separate) |

---

## Embedding Model Upgrade Path

| Phase | Model | Dims | Action |
|---|---|---|---|
| Phase 3 (current) | `all-MiniLM-L6-v2` | 384 | Baseline |
| Phase 3→4 | `multi-qa-MiniLM-L6-cos-v1` | 384 | Drop-in swap in `.env` |
| Phase 4 | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Hindi/Gujarati milestone |
| **Phase 5** ✅ | **`BAAI/bge-m3`** | **1024** | **Production. Run `make migrate-bge-m3`** |

---

## Scripts Reference

```bash
# Seed static knowledge files
python -m scripts.download_data --source knowledge

# Download all live data
python -m scripts.download_data --source all

# Run ingestion pipeline
python -m scripts.ingest

# Phase 5 migration (bge-m3)
python -m scripts.migrate_to_bge_m3 --dry-run   # preview
python -m scripts.migrate_to_bge_m3              # execute

# RAGAS evaluation
python -m scripts.evaluate --output eval_results.json
```

---

## Expected Performance (Post Phase 5)

| Query Category | Recall@5 | Notes |
|---|---|---|
| Crop disease management | 88% | bge-m3 domain fit + section chunks |
| Government scheme eligibility | 93% | Q&A pair chunks |
| Mandi commodity price | 95% | Table chunks + BM25 sparse |
| Fertiliser dose calculation | 91% | Nutrient record + tool call |
| Soil type recommendation | 87% | Soil classification chunks |
| **Hindi / Gujarati queries** | **84%** | **bge-m3 multilingual** |
| Overall (English) | 91% | — |
| End-to-end latency p95 | ~1,500 ms | Within 3s SLA |

---

## Project Structure

```
agrosight/
├── app/
│   ├── main.py                  # FastAPI app (routes, SSE)
│   ├── services/
│   │   ├── agent.py             # LangGraph ReAct agent
│   │   ├── agro_tools.py        # Weather, mandi price, fertiliser calculator
│   │   ├── chunker.py           # All chunking strategies + auto-selector
│   │   ├── embedder.py          # bge-m3 / sentence-transformers wrapper
│   │   ├── prompts.py           # System prompts, RAG templates
│   │   ├── reranker.py          # Cross-encoder reranker
│   │   ├── session_store.py     # Redis session manager
│   │   └── vector_store.py      # Qdrant client, hybrid search, RRF
│   └── utils/
│       ├── config.py            # Pydantic settings (env-driven)
│       ├── file_utils.py        # SHA-256, file iteration helpers
│       └── logger.py            # Loguru structured logging
├── scripts/
│   ├── download_data.py         # Data downloaders (weather, mandi, USDA, soil)
│   ├── ingest.py                # Full ingestion pipeline CLI
│   ├── evaluate.py              # RAGAS evaluation runner
│   └── migrate_to_bge_m3.py    # Phase 5 migration script
├── tests/
│   └── test_agrosight.py        # Pytest suite (chunker + API + prompts)
├── data/
│   ├── books/                   # Place PDF books here
│   └── raw/                     # Auto-populated by download_data.py
├── .env.example                 # Environment config template
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

---

## Configuration Reference (`.env`)

| Variable | Description | Default |
|---|---|---|
| `EMBEDDING_MODEL` | Embedding model name | `BAAI/bge-m3` |
| `EMBEDDING_DIM` | Vector dimension | `1024` |
| `RETRIEVAL_TOP_K` | Candidates from Qdrant | `8` |
| `RETRIEVAL_SCORE_THRESHOLD` | Min similarity score | `0.25` |
| `DEFAULT_CHUNK_TOKENS` | Target chunk size (words) | `512` |
| `DEFAULT_OVERLAP_TOKENS` | Chunk overlap (words) | `64` |
| `MAX_HISTORY_TURNS` | Session turns in Redis | `5` |
| `LLM_MODEL` | Ollama model name | `qwen3.5:9b` |
| `QDRANT_COLLECTION` | Vector store collection | `agricultural_knowledge_v2` |

---

## Licence
MIT — See LICENSE for details.
