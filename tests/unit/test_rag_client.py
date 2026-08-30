"""RAGClient implementations (app/clients/rag.py).

HTTPRAGClient is tested against an injected httpx.MockTransport: success,
503-not-ready, 5xx-retry, timeout, and unparseable payload. This pins the
app <-> rag wire contract without a running rag service.
"""

from __future__ import annotations

import httpx
import pytest
from app.clients.rag import HTTPRAGClient, RAGError, RAGUnavailableError, format_context
from shared.config import AppSettings
from shared.contracts import RetrievedChunk

pytestmark = pytest.mark.unit

BASE = "http://rag.test"


def _client(handler) -> HTTPRAGClient:
    settings = AppSettings(rag_base_url=BASE, rag_max_retries=2, rag_request_timeout_seconds=5.0)
    inner = httpx.Client(transport=httpx.MockTransport(handler))
    return HTTPRAGClient(settings, client=inner)


def _body(n: int = 2) -> dict:
    return {
        "chunks": [
            {"text": f"chunk {i}", "metadata": {"section": f"S{i}"}, "score": 0.5}
            for i in range(n)
        ],
        "collection": "due_diligence_knowledge",
        "query_chars": 10,
    }


def test_successful_retrieve():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/retrieve"
        return httpx.Response(200, json=_body(3))

    chunks = _client(handler).retrieve("startup info", top_k=3)
    assert len(chunks) == 3
    assert chunks[0].metadata["section"] == "S0"


def test_503_not_ready_raises_unavailable_after_retries():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="index not built")

    with pytest.raises(RAGUnavailableError):
        _client(handler).retrieve("x", top_k=5)
    assert calls["n"] == 3


def test_5xx_retry_then_success():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json=_body(1))

    chunks = _client(handler).retrieve("x", top_k=5)
    assert len(chunks) == 1
    assert calls["n"] == 2


def test_timeout_raises_unavailable():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    with pytest.raises(RAGUnavailableError):
        _client(handler).retrieve("x", top_k=5)


def test_4xx_raises_rag_error_not_retried():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, text="bad query")

    with pytest.raises(RAGError):
        _client(handler).retrieve("x", top_k=5)
    assert calls["n"] == 1


def test_unparseable_payload_raises_rag_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": 1})

    with pytest.raises(RAGError):
        _client(handler).retrieve("x", top_k=5)


def test_format_context_renders_sections():
    text = format_context(
        [
            RetrievedChunk(text="body one", metadata={"section": "Financials"}),
            RetrievedChunk(text="body two", metadata={}),
        ]
    )
    assert "[Section: Financials]" in text
    assert "[Section: Unknown section]" in text
    assert "---" in text
