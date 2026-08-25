"""ORM models for Phase 2: assessment sessions, questions (top-10 + follow-ups),
and answers.
"""

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# NOTE: every nullable column below passes an explicit SQLAlchemy type to
# mapped_column(...) instead of relying on `Mapped[X | None]` / `Mapped[Optional[X]]`
# annotation inference. SQLAlchemy 2.0.35's annotation resolver has a bug under
# Python 3.14's updated `typing` internals (TypeError in de_stringify_union_elements);
# passing the type explicitly bypasses that resolution path entirely.


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class QuestionStatus(StrEnum):
    PENDING = "PENDING"
    ANSWERED = "ANSWERED"
    SKIPPED = "SKIPPED"


class AssessmentSession(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    company_id: Mapped[str] = mapped_column(String, nullable=False)
    startup_info: Mapped[str] = mapped_column(Text, nullable=False)
    startup_stage: Mapped[str] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default=SessionStatus.IN_PROGRESS)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    questions: Mapped[list["Question"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Question.created_at"
    )
    report: Mapped["AssessmentReport"] = relationship(
        back_populates="session", uselist=False, cascade="all, delete-orphan"
    )


class Question(Base):
    __tablename__ = "questions"

    question_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), nullable=False)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=True)
    priority: Mapped[str] = mapped_column(String, nullable=True)
    reason: Mapped[str] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String, default=QuestionStatus.PENDING)

    # Top-10 questions have rank set and parent_question_id = None.
    # Follow-ups have parent_question_id pointing at the Top-10 question
    # they were generated from, and rank = None.
    rank: Mapped[int] = mapped_column(Integer, nullable=True)
    parent_question_id: Mapped[str] = mapped_column(
        String, ForeignKey("questions.question_id"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped["AssessmentSession"] = relationship(back_populates="questions")
    answer: Mapped["Answer"] = relationship(
        back_populates="question", uselist=False, cascade="all, delete-orphan"
    )
    follow_ups: Mapped[list["Question"]] = relationship(
        back_populates="parent", cascade="all, delete-orphan", order_by="Question.created_at"
    )
    parent: Mapped["Question"] = relationship(
        back_populates="follow_ups", remote_side=[question_id]
    )

    @property
    def is_followup(self) -> bool:
        return self.parent_question_id is not None

    @property
    def top_level_question_id(self) -> str:
        """The Top-10 question this question belongs to (itself, if it is one)."""
        return self.parent_question_id or self.question_id


class Answer(Base):
    __tablename__ = "answers"

    answer_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    question_id: Mapped[str] = mapped_column(
        ForeignKey("questions.question_id"), nullable=False, unique=True
    )
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), nullable=False)

    answer: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    question: Mapped["Question"] = relationship(back_populates="answer")


class AssessmentReport(Base):
    """Phase 3: the final investor due-diligence report for a completed session.

    Queryable summary fields (overall_score, risk_level, executive_summary)
    are normalized columns; the structured nested sections (strengths, risks,
    information_gaps, contradictions, category_scores, recommendations) are
    stored as JSON, mirroring the Pydantic report schema in
    app/report_engine/schemas.py exactly, so the API can hand the stored JSON
    straight back without re-deriving it.
    """

    __tablename__ = "assessment_reports"

    report_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id"), nullable=False, unique=True
    )

    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String, nullable=False)
    executive_summary: Mapped[str] = mapped_column(Text, nullable=False)

    strengths: Mapped[list] = mapped_column(JSON, nullable=False)
    risks: Mapped[list] = mapped_column(JSON, nullable=False)
    information_gaps: Mapped[list] = mapped_column(JSON, nullable=False)
    contradictions: Mapped[list] = mapped_column(JSON, nullable=False)
    category_scores: Mapped[dict] = mapped_column(JSON, nullable=False)
    recommendations: Mapped[list] = mapped_column(JSON, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped["AssessmentSession"] = relationship(back_populates="report")
