"""Live RAG pipeline smoke test: real fetch + chunk + embed + retrieve.

Replaces the old tests/test_pipeline.py Phase-1 checks that needed a built
index. Skipped unless RUN_RAG_INTEGRATION=1 (it downloads the embedding model
and fetches a live webpage).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

if os.environ.get("RUN_RAG_INTEGRATION") != "1":
    pytest.skip("set RUN_RAG_INTEGRATION=1 to run the live RAG pipeline test", allow_module_level=True)

from rag_service import service  # noqa: E402
from shared.config import get_rag_settings  # noqa: E402


def test_ingest_then_retrieve():
    settings = get_rag_settings()
    sections, chunks, _url, collection = service.ingest(
        settings, source_url=None, recreate=True
    )
    assert sections > 0 and chunks > 0

    results, name = service.retrieve(
        settings,
        query="B2B SaaS startup, 15 employees, 2 crore INR annual revenue",
        top_k=10,
        collection=None,
    )
    assert name == collection
    assert len(results) > 0
    for chunk in results:
        assert chunk.text.strip()
        assert "section" in chunk.metadata
