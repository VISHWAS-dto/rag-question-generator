"""RAG service HTTP contract (rag_service/main.py + service.py).

The vector store is faked at the `rag_service.vectorstore` boundary so these
run offline: no embedding model load, no Chroma, no network fetch. Verifies
the /retrieve and /ingest contracts and their error status codes, plus that
the shared contract models round-trip (the same types the app's HTTPRAGClient
uses).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from rag_service import service, vectorstore
from rag_service.main import app
from shared.contracts import RetrieveResponse

pytestmark = pytest.mark.unit


@pytest.fixture
def client(monkeypatch):
    # Fake the vector store layer.
    def fake_similarity_search(_settings, name, query, k):
        if name == "empty":
            raise LookupError("collection empty")
        return [
            {"text": f"chunk {i}", "metadata": {"section": f"S{i}"}, "score": 0.5}
            for i in range(min(k, 3))
        ]

    monkeypatch.setattr(vectorstore, "similarity_search", fake_similarity_search)
    monkeypatch.setattr(vectorstore, "get_embedding_model", lambda _s: object())
    monkeypatch.setattr(vectorstore, "collection_count", lambda _s, _n: 42)

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "service": "rag"}


def test_retrieve_success(client):
    resp = client.post("/retrieve", json={"query": "b2b saas startup", "top_k": 3})
    assert resp.status_code == 200
    parsed = RetrieveResponse.model_validate(resp.json())
    assert len(parsed.chunks) == 3
    assert parsed.chunks[0].metadata["section"] == "S0"


def test_retrieve_empty_query_is_422(client):
    assert client.post("/retrieve", json={"query": "", "top_k": 3}).status_code == 422


def test_retrieve_top_k_out_of_bounds_is_422(client):
    assert client.post("/retrieve", json={"query": "x", "top_k": 999}).status_code == 422


def test_retrieve_missing_index_is_503(client):
    resp = client.post(
        "/retrieve", json={"query": "x", "top_k": 3, "collection": "empty"}
    )
    assert resp.status_code == 503


def test_ready_reports_checks(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert set(resp.json()["checks"]) == {"embedding_model", "vector_store"}


def test_ingest_failure_is_502(client, monkeypatch):
    def boom(*_a, **_k):
        raise service.IngestionFailed("source unreachable")

    monkeypatch.setattr(service, "ingest", boom)
    resp = client.post("/ingest", json={"recreate": True})
    assert resp.status_code == 502
