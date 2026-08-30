"""Shared test fixtures.

Everything here is offline and deterministic. The LLM and RAG are replaced at
their client seams (`FakeLLMClient`, `FakeRAGClient`), the DB is in-memory
SQLite, and settings are forced to the `test` environment. No fixture hits the
network or the real vector store.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from app.clients.llm import FakeLLMClient
from app.clients.rag import FakeRAGClient
from app.domain.schemas import Category
from app.orchestration.interview_service import InterviewService
from app.persistence.models import (
    Answer,
    AssessmentSession,
    Base,
    Question,
    QuestionStatus,
    SessionStatus,
)
from shared.config import AppSettings, Environment, LLMProvider, RagMode
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

STARTUP_INFO = (
    "We're a B2B SaaS startup, 15 employees, 2 crore INR annual revenue. "
    "70% of our revenue comes from our top five customers."
)


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        environment=Environment.TEST,
        llm_provider=LLMProvider.FAKE,
        rag_mode=RagMode.FAKE,
        database_url="sqlite+pysqlite:///:memory:",
        readiness_check_dependencies=False,
        num_questions=3,
        max_followups_per_question=3,
        log_json=False,
    )


@pytest.fixture
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_factory(db_engine) -> sessionmaker[Session]:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest.fixture
def db(db_factory) -> Iterator[Session]:
    session = db_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def fake_rag() -> FakeRAGClient:
    return FakeRAGClient()


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def service(fake_llm, fake_rag, settings) -> InterviewService:
    return InterviewService(llm=fake_llm, rag=fake_rag, settings=settings)


# --------------------------------------------------------------------------- #
# Canned LLM payload helpers
# --------------------------------------------------------------------------- #


def top_questions_payload(n: int = 3) -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "question": f"Seeded question {i}: explain an aspect of the business.",
                    "category": "Financial",
                    "priority": "High",
                    "reason": "seeded",
                    "source_context": "Financial Diligence",
                }
                for i in range(1, n + 1)
            ]
        }
    )


def followup_payload(
    question: str,
    *,
    category: str = "Financial",
    priority: str = "High",
    reason: str = "probe deeper",
) -> str:
    return json.dumps(
        {
            "follow_up_required": True,
            "question": question,
            "category": category,
            "priority": priority,
            "reason": reason,
        }
    )


def no_followup_payload(reason: str = "Answer is sufficient.") -> str:
    return json.dumps(
        {
            "follow_up_required": False,
            "question": None,
            "category": None,
            "priority": None,
            "reason": reason,
        }
    )


def analysis_payload(**overrides) -> str:
    base = {
        "executive_summary": "Early-stage B2B SaaS with meaningful customer concentration risk.",
        "strengths": [],
        "risks": [],
        "information_gaps": [],
        "contradictions": [],
        "category_assessments": [
            {
                "category": c.value,
                "assessment": "Moderate",
                "rationale": "Limited detail provided.",
                "evidence_strength": "Low",
                "evidence_gaps": ["quantitative detail"],
            }
            for c in Category
        ],
        "recommendations": [],
    }
    base.update(overrides)
    return json.dumps(base)


# --------------------------------------------------------------------------- #
# DB builders
# --------------------------------------------------------------------------- #


def make_session_with_questions(
    db: Session, questions: list[tuple[str, str]] | None = None
) -> AssessmentSession:
    session = AssessmentSession(
        company_id="acme-co",
        startup_info=STARTUP_INFO,
        startup_stage="Seed",
        status=SessionStatus.IN_PROGRESS,
    )
    db.add(session)
    db.flush()

    questions = questions or [
        ("What percentage of revenue comes from your top five customers?", "Financial"),
        ("What is your current customer retention rate?", "Traction"),
        ("How many engineers are on your team?", "Team"),
    ]
    for rank, (text, category) in enumerate(questions, start=1):
        db.add(
            Question(
                session_id=session.session_id,
                question=text,
                category=category,
                priority="High",
                reason="seeded",
                status=QuestionStatus.PENDING,
                rank=rank,
            )
        )
    db.commit()
    db.refresh(session)
    return session


def make_completed_session(
    db: Session, *, answer: str = "5 crore INR annual revenue.", second_answer: str | None = None
) -> AssessmentSession:
    session = AssessmentSession(
        company_id="acme-co",
        startup_info=STARTUP_INFO,
        startup_stage="Seed",
        status=SessionStatus.IN_PROGRESS,
    )
    db.add(session)
    db.flush()

    q1 = Question(
        session_id=session.session_id,
        question="What is your annual revenue?",
        category="Financial",
        priority="High",
        reason="seeded",
        status=QuestionStatus.PENDING,
        rank=1,
    )
    db.add(q1)
    db.flush()
    db.add(Answer(question_id=q1.question_id, session_id=session.session_id, answer=answer))
    q1.status = QuestionStatus.ANSWERED

    if second_answer is not None:
        q2 = Question(
            session_id=session.session_id,
            question="Can you confirm that revenue figure again?",
            category="Financial",
            priority="High",
            reason="seeded",
            status=QuestionStatus.PENDING,
            rank=2,
        )
        db.add(q2)
        db.flush()
        db.add(
            Answer(question_id=q2.question_id, session_id=session.session_id, answer=second_answer)
        )
        q2.status = QuestionStatus.ANSWERED

    db.commit()
    db.refresh(session)
    return session
