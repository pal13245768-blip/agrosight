# =============================================================================
#  AgroSight – Developer Makefile
# =============================================================================

.PHONY: help install dev-server docker-up docker-down download-data ingest evaluate test clean logs

PYTHON := python
PIP    := pip

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────────────

install:  ## Install Python dependencies
	$(PIP) install -r requirements.txt

setup-env:  ## Copy .env.example to .env (edit before running)
	cp -n .env.example .env && echo "✅  .env created. Fill in API keys before proceeding."

# ── Development ────────────────────────────────────────────────────────────

dev-server:  ## Start FastAPI dev server with auto-reload
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# ── Docker ─────────────────────────────────────────────────────────────────

docker-up:  ## Start API + Redis in Docker
	docker compose up -d --build
	@echo "✅  AgroSight API running at http://localhost:8000"
	@echo "📖  Docs at http://localhost:8000/docs"

docker-down:  ## Stop Docker services
	docker compose down

docker-logs:  ## Tail API container logs
	docker compose logs -f api

# ── Data pipeline ──────────────────────────────────────────────────────────

download-knowledge:  ## Seed static knowledge JSON files (no API key needed)
	$(PYTHON) -m scripts.download_data --source knowledge

download-weather:  ## Download weather advisories (needs OPENWEATHER_API_KEY)
	$(PYTHON) -m scripts.download_data --source weather

download-mandi:  ## Download mandi prices (needs DATA_GOV_API_KEY_1)
	$(PYTHON) -m scripts.download_data --source mandi

download-all:  ## Download all data sources
	$(PYTHON) -m scripts.download_data --source all

ingest:  ## Run full ingestion pipeline (chunk + embed + upsert to Qdrant)
	$(PYTHON) -m scripts.ingest

ingest-fresh:  ## Recreate Qdrant collection and re-ingest (Phase 5 migration)
	$(PYTHON) -m scripts.ingest --recreate-collection

migrate-bge-m3:  ## Phase 5: migrate to BAAI/bge-m3 (dry run first)
	$(PYTHON) -m scripts.migrate_to_bge_m3 --dry-run
	@echo "Remove --dry-run flag to execute: python -m scripts.migrate_to_bge_m3"

# ── Evaluation ─────────────────────────────────────────────────────────────

evaluate:  ## Run RAGAS evaluation with built-in eval set
	$(PYTHON) -m scripts.evaluate --output eval_results.json
	@echo "Results saved to eval_results.json"

# ── Testing ────────────────────────────────────────────────────────────────

test:  ## Run unit + integration tests
	pytest tests/ -v --tb=short

test-chunker:  ## Run chunker unit tests only
	pytest tests/test_agrosight.py -v -k "Chunk or chunk or Semantic or Section or QA or Table or Record or Sliding"

test-api:  ## Run API endpoint tests only
	pytest tests/test_agrosight.py -v -k "API"

# ── Utilities ──────────────────────────────────────────────────────────────

clean:  ## Remove Python cache and compiled files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

logs:  ## Tail application log file
	tail -f logs/agrosight.log

check-health:  ## Check API health endpoint
	curl -s http://localhost:8000/health | python -m json.tool

# ── Quick start ────────────────────────────────────────────────────────────

quickstart: setup-env install download-knowledge ingest dev-server  ## Full local setup from scratch
	@echo "🌾  AgroSight is running!"
