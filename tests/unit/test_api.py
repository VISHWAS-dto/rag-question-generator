"""API contract tests via FastAPI TestClient.

The app is built with fake LLM/RAG clients and an in-memory DB by overriding
the startup-wired state. Verifies status-code mapping for the full happy path
and the important error paths.
"""

from __future__ import annotations

import pytest
from app.api.dependencies import get_interview_service
from app.api.main import create_app
from app.clients.llm import FakeLLMClient
from app.clients.rag import FakeRAGClient
from app.orchestration.interview_service import InterviewService
from app.persistence.database import get_session
from app.persistence.models import Base
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from tests.conftest import (
    analysis_payload,
    no_followup_payload,
    top_questions_payload,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def client_and_llm(settings):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, expire_on_commit=False)

    llm = FakeLLMClient(
        responses=[
            top_questions_payload(3),
            no_followup_payload(),
            no_followup_payload(),
            no_followup_payload(),
            analysis_payload(),
        ]
    )
    rag = FakeRAGClient()
    svc = InterviewService(llm=llm, rag=rag, settings=settings)

    app = create_app(settings)

    def _override_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = _override_db
    app.dependency_overrides[get_interview_service] = lambda: svc

    with TestClient(app) as client:
        yield client, llm


def test_full_happy_path(client_and_llm):
    client, _ = client_and_llm

    resp = client.post(
        "/sessions",
        json={"company_id": "acme", "startup_info": "B2B SaaS, 15 staff.", "startup_stage": "Seed"},
    )
    assert resp.status_code == 201, resp.text
    session_id = resp.json()["session_id"]

    resp = client.get(f"/sessions/{session_id}/questions")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["questions"]) == 3
    current = body["current_question"]

    # Answer all three; the last completes the session.
    qid = current["question_id"]
    for _ in range(3):
        resp = client.post(f"/questions/{qid}/answer", json={"answer": "A concrete answer."})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if data["type"] == "complete":
            break
        qid = data["question"]["question_id"]

    resp = client.post(f"/sessions/{session_id}/complete")
    assert resp.status_code == 200, resp.text
    report_id = resp.json()["report_id"]

    # Idempotent
    resp2 = client.post(f"/sessions/{session_id}/complete")
    assert resp2.status_code == 200
    assert resp2.json()["report_id"] == report_id

    resp3 = client.get(f"/sessions/{session_id}/report")
    assert resp3.status_code == 200
    assert resp3.json()["report_id"] == report_id


def test_create_session_validation_error_is_422(client_and_llm):
    client, _ = client_and_llm
    resp = client.post("/sessions", json={"company_id": "", "startup_info": "x"})
    assert resp.status_code == 422


def test_missing_field_is_422(client_and_llm):
    client, _ = client_and_llm
    resp = client.post("/sessions", json={"company_id": "acme"})
    assert resp.status_code == 422


def test_oversized_startup_info_is_422(client_and_llm):
    client, _ = client_and_llm
    resp = client.post(
        "/sessions", json={"company_id": "acme", "startup_info": "x" * 9000}
    )
    assert resp.status_code == 422


def test_unknown_session_is_404(client_and_llm):
    client, _ = client_and_llm
    assert client.get("/sessions/nope").status_code == 404
    assert client.get("/sessions/nope/questions").status_code == 404
    assert client.get("/sessions/nope/report").status_code == 404


def test_complete_incomplete_session_is_409(client_and_llm):
    client, llm = client_and_llm
    llm.responses.insert(0, top_questions_payload(3))
    resp = client.post(
        "/sessions", json={"company_id": "acme", "startup_info": "info here"}
    )
    session_id = resp.json()["session_id"]
    resp = client.post(f"/sessions/{session_id}/complete")
    assert resp.status_code == 409


def test_answer_unknown_question_is_404(client_and_llm):
    client, _ = client_and_llm
    resp = client.post("/questions/nope/answer", json={"answer": "hi"})
    assert resp.status_code == 404


def test_empty_answer_is_422(client_and_llm):
    client, _ = client_and_llm
    resp = client.post("/questions/whatever/answer", json={"answer": ""})
    assert resp.status_code == 422


def test_health_and_ready(client_and_llm):
    client, _ = client_and_llm
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["checks"]["database"] == "ok"


def test_request_id_header_echoed(client_and_llm):
    client, _ = client_and_llm
    r = client.get("/health", headers={"x-request-id": "test-req-1"})
    assert r.headers["x-request-id"] == "test-req-1"
