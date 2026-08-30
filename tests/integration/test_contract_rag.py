"""Contract test: the app's HTTPRAGClient and the rag service agree on the
wire format.

The real rag_service FastAPI app runs in-process via TestClient (vector store
faked). A MockTransport bridges a real HTTPRAGClient to that TestClient, so
both sides exercise their actual `shared.contracts` usage. If either drifts,
this breaks.
"""

from __future__ import annotations

import httpx
import pytest
from app.clients.rag import HTTPRAGClient
from fastapi.testclient import TestClient
from rag_service import vectorstore
from rag_service.main import app as rag_app
from shared.config import AppSettings

pytestmark = pytest.mark.integration


@pytest.fixture
def rag_client(monkeypatch):
    monkeypatch.setattr(
        vectorstore,
        "similarity_search",
        lambda _s, _n, _q, k: [
            {"text": f"c{i}", "metadata": {"section": f"S{i}", "chunk_index": i}, "score": 0.4}
            for i in range(min(k, 4))
        ],
    )
    monkeypatch.setattr(vectorstore, "get_embedding_model", lambda _s: object())
    monkeypatch.setattr(vectorstore, "collection_count", lambda _s, _n: 10)

    server = TestClient(rag_app)

    def bridge(request: httpx.Request) -> httpx.Response:
        raw = server.request(
            request.method,
            request.url.path,
            content=request.content,
            headers=dict(request.headers),
            params=dict(request.url.params),
        )
        return httpx.Response(raw.status_code, content=raw.content, headers=dict(raw.headers))

    inner = httpx.Client(transport=httpx.MockTransport(bridge), base_url="http://rag.local")
    settings = AppSettings(rag_base_url="http://rag.local", rag_max_retries=0)
    with server:
        yield HTTPRAGClient(settings, client=inner)
    inner.close()


def test_retrieve_round_trips(rag_client):
    chunks = rag_client.retrieve("a b2b saas startup with 10k customers", top_k=4)
    assert len(chunks) == 4
    assert chunks[0].metadata["section"] == "S0"
    assert chunks[0].score == pytest.approx(0.4)


def test_health_round_trips(rag_client):
    assert rag_client.health() is True
