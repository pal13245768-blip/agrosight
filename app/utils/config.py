"""
AgroSight – Centralised application configuration.
All values are loaded from environment variables (via .env).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────
    app_title: str = "AgroSight RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    request_timeout: int = 30
    max_agent_iterations: int = 6

    # ── LLM (Mistral API) ─────────────────────────────────────────────────
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")
    mistral_model: str = "mistral-large-latest"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # ── Qdrant ────────────────────────────────────────────────────────────
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: str = Field(..., env="QDRANT_API_KEY")
    qdrant_collection: str = "agricultural_knowledge_v2"

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieval_top_k: int = 8
    retrieval_score_threshold: float = 0.18
    default_chunk_tokens: int = 512
    default_overlap_tokens: int = 64

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 86_400
    max_history_turns: int = 5

    # ── Weather ───────────────────────────────────────────────────────────
    openweather_api_key: str = ""
    openweather_base_url: str = "https://api.openweathermap.org/data/2.5"

    # ── Government data ───────────────────────────────────────────────────
    data_gov_api_key_1: str = ""
    data_gov_base_url: str = "https://www.data.gov.in/api"
    data_gov_mandi_resource_id: str = "35985678-0d79-46b4-9ed6-6f13308a1d24"

    agmarknet_api_key: str = ""
    agmarknet_base_url: str = "https://agmarknet.gov.in/api"

    usda_nass_api_key: str = ""
    usda_nass_base_url: str = "https://quickstats.nass.usda.gov/api"

    fao_faostat_base_url: str = "https://fenixservices.fao.org/faostat/api/v1"
    isric_base_url: str = "https://rest.isric.org/soilgrids/v2.0"

    # ── Kaggle ────────────────────────────────────────────────────────────
    kaggle_username: str = ""
    kaggle_key: str = ""

    # ── Data paths ────────────────────────────────────────────────────────
    data_root: str = "./data/raw"
    books_dir: str = "./data/books"
    chunks_output_dir: str = "./chunks_output"

    # ── Ingestion ─────────────────────────────────────────────────────────
    max_retries: int = 3
    retry_backoff: int = 2
    batch_size: int = 100

    @field_validator("log_level", mode="before")
    @classmethod
    def normalise_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
