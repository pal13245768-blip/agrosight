# =============================================================================
#  AgroSight – Dockerfile
#  Multi-stage build: deps → runtime
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for PDF processing and ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-guj \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-guj \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY .env.example .env.example

# Create runtime directories
RUN mkdir -p logs chunks_output data/raw data/books

# Non-root user for security
RUN useradd -m -u 1000 agrosight && chown -R agrosight:agrosight /app
USER agrosight

EXPOSE 8000

# Pre-download embedding model cache at build time (optional — comment out to skip)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--timeout-keep-alive", "30"]
