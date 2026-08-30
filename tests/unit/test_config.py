"""Settings behave per-environment and validate input (shared/config.py)."""

from __future__ import annotations

import pytest
from shared.config import AppSettings, Environment, LLMProvider, RagMode, RagSettings

pytestmark = pytest.mark.unit


def test_cors_wildcard_and_list_parsing():
    assert AppSettings(cors_allow_origins="*").cors_origins_list == ["*"]
    s = AppSettings(cors_allow_origins="https://a.example, https://b.example")
    assert s.cors_origins_list == ["https://a.example", "https://b.example"]


def test_empty_cors_falls_back_to_wildcard():
    assert AppSettings(cors_allow_origins="   ").cors_origins_list == ["*"]


def test_is_sqlite_detection():
    assert AppSettings(database_url="sqlite+pysqlite:///./x.db").is_sqlite is True
    assert (
        AppSettings(database_url="postgresql+psycopg://u:p@h/db").is_sqlite is False
    )


def test_env_prefixes_are_isolated(monkeypatch):
    monkeypatch.setenv("APP_LLM_PROVIDER", "openai_compat")
    monkeypatch.setenv("APP_RAG_MODE", "in_process")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "512")
    app = AppSettings()
    rag = RagSettings()
    assert app.llm_provider == LLMProvider.OPENAI_COMPAT
    assert app.rag_mode == RagMode.IN_PROCESS
    assert rag.chunk_size == 512


def test_invalid_chroma_mode_rejected():
    with pytest.raises(ValueError, match="RAG_CHROMA_MODE"):
        RagSettings(chroma_mode="lmdb")


def test_is_production_flag():
    assert AppSettings(environment=Environment.PRODUCTION).is_production is True
    assert AppSettings(environment=Environment.LOCAL).is_production is False
