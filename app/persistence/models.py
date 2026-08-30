"""ORM models: assessment sessions, questions (top-10 + follow-ups), answers,
and the final report.

Ported from the original app/models.py. Column types are passed explicitly
(not inferred from `Mapped[X | None]`) to stay compatible across SQLAlchemy /
Python typing versions. An `idempotency_key` was added to sessions so a
retried "create session" call does not spawn a duplicate interview.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class SessionStatus(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class QuestionStatus(StrEnum):
    PENDING = "PENDING"
    ANSWERED = "ANSWERED"
    SKIPPED = "SKIPPED"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AssessmentSession(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_sessions_idempotency_key"),
    )

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    company_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    startup_info: Mapped[str] = mapped_column(Text, nullable=False)
    startup_stage: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default=SessionStatus.IN_PROGRESS, nullable=False
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    questions: Mapped[list[Question]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Question.created_at",
    )
    report: Mapped[AssessmentReport | None] = relationship(
        back_populates="session", uselist=False, cascade="all, delete-orphan"
    )


class Question(Base):
    __tablename__ = "questions"

    question_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id"), nullable=False, index=True
    )

    question: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    priority: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(
        String(32), default=QuestionStatus.PENDING, nullable=False
    )

    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parent_question_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("questions.question_id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    session: Mapped[AssessmentSession] = relationship(back_populates="questions")
    answer: Mapped[Answer | None] = relationship(
        back_populates="question", uselist=False, cascade="all, delete-orphan"
    )
    follow_ups: Mapped[list[Question]] = relationship(
        back_populates="parent",
        cascade="all, delete-orphan",
        order_by="Question.created_at",
    )
    parent: Mapped[Question | None] = relationship(
        back_populates="follow_ups", remote_side=[question_id]
    )

    @property
    def is_followup(self) -> bool:
        return self.parent_question_id is not None

    @property
    def top_level_question_id(self) -> str:
        return self.parent_question_id or self.question_id


class Answer(Base):
    __tablename__ = "answers"

    answer_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    question_id: Mapped[str] = mapped_column(
        ForeignKey("questions.question_id"), nullable=False, unique=True
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id"), nullable=False, index=True
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    question: Mapped[Question] = relationship(back_populates="answer")


class AssessmentReport(Base):
    __tablename__ = "assessment_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id"), nullable=False, unique=True
    )

    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(32), nullable=False)
    executive_summary: Mapped[str] = mapped_column(Text, nullable=False)

    strengths: Mapped[list] = mapped_column(JSON, nullable=False)
    risks: Mapped[list] = mapped_column(JSON, nullable=False)
    information_gaps: Mapped[list] = mapped_column(JSON, nullable=False)
    contradictions: Mapped[list] = mapped_column(JSON, nullable=False)
    category_scores: Mapped[dict] = mapped_column(JSON, nullable=False)
    recommendations: Mapped[list] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    session: Mapped[AssessmentSession] = relationship(back_populates="report")
