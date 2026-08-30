"""API routes. Maps the interview state machine to REST, and every typed
orchestration error to the correct HTTP status in one exception handler.

Endpoints (unchanged contract from the original app):
    POST /sessions                      create session + seed questions
    GET  /sessions/{id}                 session status
    GET  /sessions/{id}/questions       all questions + the current one
    POST /questions/{id}/answer         submit an answer -> follow-up / next / complete
    POST /sessions/{id}/complete        generate the report (idempotent)
    GET  /sessions/{id}/report          fetch the stored report
"""

from __future__ import annotations

from typing import NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session as DBSession

from app.api.dependencies import get_interview_service
from app.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    AssessmentReportSchema,
    CreateSessionRequest,
    QuestionResponse,
    SessionQuestionsResponse,
    SessionResponse,
)
from app.orchestration.errors import (
    ConflictError,
    DependencyUnavailableError,
    NotFoundError,
    UpstreamError,
    UpstreamTimeoutError,
    ValidationError,
)
from app.orchestration.interview_service import InterviewService
from app.persistence.database import get_session
from app.persistence.models import Question, QuestionStatus

router = APIRouter()


def _q(q: Question) -> QuestionResponse:
    return QuestionResponse(
        question_id=q.question_id,
        session_id=q.session_id,
        question=q.question,
        category=q.category,
        priority=q.priority,
        reason=q.reason,
        status=q.status,
        is_followup=q.is_followup,
        parent_question_id=q.parent_question_id,
        created_at=q.created_at,
    )


def _raise_http(exc: Exception) -> NoReturn:
    """Translate an orchestration error into an HTTPException (always raises)."""
    mapping: list[tuple[type, int]] = [
        (NotFoundError, 404),
        (ValidationError, 422),
        (ConflictError, 409),
        (UpstreamTimeoutError, 504),
        (DependencyUnavailableError, 503),
        (UpstreamError, 502),
    ]
    for err_type, status in mapping:
        if isinstance(exc, err_type):
            raise HTTPException(status_code=status, detail=str(exc)) from exc
    raise exc


@router.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(
    body: CreateSessionRequest,
    svc: InterviewService = Depends(get_interview_service),
    db: DBSession = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> SessionResponse:
    try:
        session = svc.create_session(
            db,
            company_id=body.company_id,
            startup_info=body.startup_info,
            startup_stage=body.startup_stage,
            idempotency_key=idempotency_key,
        )
    except Exception as exc:
        _raise_http(exc)
    return SessionResponse(
        session_id=session.session_id,
        company_id=session.company_id,
        startup_stage=session.startup_stage,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session_endpoint(
    session_id: str,
    svc: InterviewService = Depends(get_interview_service),
    db: DBSession = Depends(get_session),
) -> SessionResponse:
    try:
        session = svc.get_session(db, session_id)
    except Exception as exc:
        _raise_http(exc)
    return SessionResponse(
        session_id=session.session_id,
        company_id=session.company_id,
        startup_stage=session.startup_stage,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/sessions/{session_id}/questions", response_model=SessionQuestionsResponse)
def get_session_questions(
    session_id: str,
    svc: InterviewService = Depends(get_interview_service),
    db: DBSession = Depends(get_session),
) -> SessionQuestionsResponse:
    try:
        questions = svc.list_questions(db, session_id)
        current = svc.get_current_question(db, session_id)
    except Exception as exc:
        _raise_http(exc)
    return SessionQuestionsResponse(
        questions=[_q(q) for q in questions],
        current_question=_q(current) if current else None,
    )


@router.post("/questions/{question_id}/answer", response_model=AnswerResponse)
def submit_answer(
    question_id: str,
    body: AnswerRequest,
    svc: InterviewService = Depends(get_interview_service),
    db: DBSession = Depends(get_session),
) -> AnswerResponse:
    try:
        result = svc.submit_answer(db, question_id=question_id, answer_text=body.answer)
    except Exception as exc:
        _raise_http(exc)

    if result.session_completed:
        response_type = "complete"
    elif result.decision.follow_up_required:
        response_type = "follow_up"
    else:
        response_type = "next_question"

    return AnswerResponse(
        type=response_type,
        follow_up_required=result.decision.follow_up_required,
        reason=result.decision.reason,
        question=_q(result.next_question) if result.next_question else None,
    )


@router.post("/sessions/{session_id}/complete", response_model=AssessmentReportSchema)
def complete_session(
    session_id: str,
    svc: InterviewService = Depends(get_interview_service),
    db: DBSession = Depends(get_session),
) -> AssessmentReportSchema:
    try:
        return svc.complete_session(db, session_id)
    except Exception as exc:
        _raise_http(exc)


@router.get("/sessions/{session_id}/report", response_model=AssessmentReportSchema)
def get_session_report(
    session_id: str,
    svc: InterviewService = Depends(get_interview_service),
    db: DBSession = Depends(get_session),
) -> AssessmentReportSchema:
    try:
        return svc.get_report(db, session_id)
    except Exception as exc:
        _raise_http(exc)


# Keep QuestionStatus importable from here for tests that used the old module layout.
__all__ = ["QuestionStatus", "router"]
