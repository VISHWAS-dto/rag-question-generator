"""Request/response models for the HTTP API.

Validation limits (field lengths) are enforced here so malformed input is
rejected at the edge with 422 and a precise message, before any DB or LLM work.
The hard caps mirror `AppSettings`; Field constraints use conservative
constants so the schema is self-documenting in OpenAPI.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.schemas import AssessmentReportSchema

__all__ = [
    "AnswerRequest",
    "AnswerResponse",
    "AssessmentReportSchema",
    "CreateSessionRequest",
    "ErrorResponse",
    "QuestionResponse",
    "SessionQuestionsResponse",
    "SessionResponse",
]


class CreateSessionRequest(BaseModel):
    company_id: str = Field(min_length=1, max_length=128)
    startup_info: str = Field(min_length=1, max_length=8_000)
    startup_stage: str | None = Field(default=None, max_length=128)


class SessionResponse(BaseModel):
    session_id: str
    company_id: str
    startup_stage: str | None
    status: str
    created_at: datetime
    updated_at: datetime


class QuestionResponse(BaseModel):
    question_id: str
    session_id: str
    question: str
    category: str | None
    priority: str | None
    reason: str | None
    status: str
    is_followup: bool
    parent_question_id: str | None
    created_at: datetime


class SessionQuestionsResponse(BaseModel):
    questions: list[QuestionResponse]
    current_question: QuestionResponse | None


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=8_000)


class AnswerResponse(BaseModel):
    type: str  # "follow_up" | "next_question" | "complete"
    follow_up_required: bool
    reason: str
    question: QuestionResponse | None


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
