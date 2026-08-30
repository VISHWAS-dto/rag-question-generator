"""Environment-driven configuration for both services.

All configuration comes from environment variables (or a local `.env` file
during development). Nothing is hardcoded per environment: the same container
images run in local, staging, and production, and only the environment
differs. Settings are validated at startup — a missing or malformed value
fails fast with a clear error instead of surfacing as a runtime 500 later.

Three settings classes:
  * `AppSettings`  - the `app` service (API, orchestration, persistence, clients)
  * `RagSettings`  - the `rag` service (embeddings, vector store, retrieval)
  * `CommonSettings` - fields both share (logging, environment name)

Use the module-level `get_app_settings()` / `get_rag_settings()` accessors so
settings are constructed once and dependency-injected, never imported as
module-level constants.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LLMProvider(StrEnum):
    """Which `LLMClient` implementation to construct.

    `openai_compat` covers self-hosted vLLM, NVIDIA's OpenAI-compatible
    endpoint, and OpenAI itself - they share a wire format, so only the base
    URL / key / model name change. `echo` is a dependency-free stub for local
    dev and CI. `fake` is wired only by the test suite.
    """

    OPENAI_COMPAT = "openai_compat"
    ECHO = "echo"
    FAKE = "fake"


class RagMode(StrEnum):
    """How the `app` service reaches retrieval.

    `http` calls the standalone `rag` service (production). `in_process` keeps
    retrieval in the app's own process (single-node / low-footprint mode).
    `fake` is wired only by the test suite.
    """

    HTTP = "http"
    IN_PROCESS = "in_process"
    FAKE = "fake"


_ENV_FILE = (".env", ".env.local")


class CommonSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    environment: Environment = Environment.LOCAL
    log_level: str = "INFO"
    # JSON logs everywhere except explicitly-human local runs.
    log_json: bool = True

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


class AppSettings(CommonSettings):
    """Configuration for the `app` service."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_prefix="APP_", extra="ignore", case_sensitive=False
    )

    service_name: str = "app"
    host: str = "0.0.0.0"
    port: int = 8000

    # --- CORS: comma-separated origins, or "*" ---
    cors_allow_origins: str = "*"

    # --- Request limits ---
    max_startup_info_chars: int = 8_000
    max_answer_chars: int = 8_000
    max_company_id_chars: int = 128
    max_request_body_bytes: int = 256 * 1024

    # --- Database ---
    # SQLAlchemy URL. Default is a local SQLite file so `uv run` works with no
    # infra; production sets this to a postgresql+psycopg:// URL.
    database_url: str = "sqlite+pysqlite:///./data/app.db"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout_seconds: int = 30
    db_pool_recycle_seconds: int = 1800

    # --- LLM client ---
    llm_provider: LLMProvider = LLMProvider.ECHO
    llm_base_url: str = "http://llm:8000/v1"
    llm_api_key: str = "not-needed-for-self-hosted"
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_temperature: float = 0.4
    llm_request_timeout_seconds: float = 180.0
    llm_connect_timeout_seconds: float = 10.0
    llm_max_retries: int = 2
    # Default output ceiling; the report step overrides this per-call.
    llm_max_tokens: int = 1024
    llm_report_max_tokens: int = 4096
    # Repair loop: re-invocations allowed after the first attempt.
    llm_max_repair_attempts: int = 2

    # --- RAG client ---
    rag_mode: RagMode = RagMode.HTTP
    rag_base_url: str = "http://rag:8100"
    rag_request_timeout_seconds: float = 20.0
    rag_connect_timeout_seconds: float = 5.0
    rag_max_retries: int = 2
    rag_retrieval_top_k: int = 10
    rag_followup_top_k: int = 4
    rag_collection: str = "due_diligence_knowledge"

    # --- Interview engine ---
    num_questions: int = 10
    max_followups_per_question: int = 3

    # --- Readiness ---
    # Whether /ready should also probe the llm and rag services. Off in local
    # echo mode so `uv run` reports ready without a full stack.
    readiness_check_dependencies: bool = True

    @field_validator("cors_allow_origins")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return v.strip() or "*"

    @property
    def cors_origins_list(self) -> list[str]:
        raw = self.cors_allow_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


class RagSettings(CommonSettings):
    """Configuration for the `rag` service."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_prefix="RAG_", extra="ignore", case_sensitive=False
    )

    service_name: str = "rag"
    host: str = "0.0.0.0"
    port: int = 8100

    # --- Source document for the knowledge base ---
    source_url: str = (
        "https://www.startupscience.io/articles/startup-due-diligence-checklist"
    )
    fetch_timeout_seconds: int = 15

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Embeddings (local, CPU, no API key) ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store ---
    # "embedded" uses a local persistent Chroma directory; "http" connects to a
    # standalone Chroma server (chromadb/chroma image) for shared state across
    # rag replicas.
    chroma_mode: str = "embedded"
    chroma_persist_dir: str = "./data/chroma"
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    chroma_collection: str = "due_diligence_knowledge"

    # --- Retrieval defaults ---
    default_top_k: int = 10
    max_top_k: int = 50

    @field_validator("chroma_mode")
    @classmethod
    def _valid_chroma_mode(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"embedded", "http"}:
            raise ValueError("RAG_CHROMA_MODE must be 'embedded' or 'http'")
        return v


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings()


@lru_cache
def get_rag_settings() -> RagSettings:
    return RagSettings()


def reset_settings_cache() -> None:
    """Test hook: drop cached settings so a test can re-read the environment."""
    get_app_settings.cache_clear()
    get_rag_settings.cache_clear()
