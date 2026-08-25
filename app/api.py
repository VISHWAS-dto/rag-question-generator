"""Phase 2 FastAPI endpoints: sessions, questions, answers, follow-ups."""

from datetime import datetime

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from app.db import get_db, init_db
from app.models import Question, QuestionStatus
from app.report_engine.report_generator import ReportGenerationError
from app.report_engine.schemas import AssessmentReportSchema
from app.session_manager import (
    IncompleteSessionError,
    NotFoundError,
    ValidationError,
    complete_session,
    create_session,
    get_current_question,
    get_question,
    get_report,
    get_session,
    list_questions,
    submit_answer,
)

router = APIRouter()


# --- Request/response schemas ---


class CreateSessionRequest(BaseModel):
    company_id: str
    startup_info: str
    startup_stage: str | None = None


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


class AnswerRequest(BaseModel):
    answer: str


class AnswerResponse(BaseModel):
    type: str  # "follow_up" | "next_question" | "complete"
    follow_up_required: bool
    reason: str
    question: QuestionResponse | None


def _question_to_response(q: Question) -> QuestionResponse:
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


@router.post("/sessions", response_model=SessionResponse)
def post_session(body: CreateSessionRequest, db: DBSession = Depends(get_db)):
    try:
        session = create_session(
            db,
            company_id=body.company_id,
            startup_info=body.startup_info,
            startup_stage=body.startup_stage,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return session


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session_endpoint(session_id: str, db: DBSession = Depends(get_db)):
    try:
        return get_session(db, session_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


class SessionQuestionsResponse(BaseModel):
    questions: list[QuestionResponse]
    current_question: QuestionResponse | None


@router.get("/sessions/{session_id}/questions", response_model=SessionQuestionsResponse)
def get_session_questions(session_id: str, db: DBSession = Depends(get_db)):
    """Lists all questions asked so far in the session (top-10 + follow-ups),
    plus current_question: the next one the user should answer (the first
    pending top-10 question at the very start of a session).
    """
    try:
        questions = list_questions(db, session_id)
        current = get_current_question(db, session_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SessionQuestionsResponse(
        questions=[_question_to_response(q) for q in questions],
        current_question=_question_to_response(current) if current else None,
    )


@router.post("/questions/{question_id}/answer", response_model=AnswerResponse)
def post_answer(question_id: str, body: AnswerRequest, db: DBSession = Depends(get_db)):
    try:
        result = submit_answer(db, question_id, body.answer)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
        question=_question_to_response(result.next_question) if result.next_question else None,
    )


@router.post("/questions/{question_id}/follow-up", response_model=AnswerResponse)
def post_followup(question_id: str, db: DBSession = Depends(get_db)):
    """Re-run follow-up analysis for a question that has already been
    answered, without submitting a new answer. Useful for inspecting or
    re-triggering the follow-up decision independently of the answer flow.
    """
    try:
        question = get_question(db, question_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if question.status != QuestionStatus.ANSWERED or question.answer is None:
        raise HTTPException(
            status_code=422, detail=f"Question {question_id} has not been answered yet"
        )

    from app.question_engine.context_builder import build_followup_context
    from app.question_engine.followup import decide_followup

    ctx = build_followup_context(question.session, question, question.answer.answer)
    decision = decide_followup(ctx)

    next_question = None
    if decision.follow_up_required and decision.question:
        next_question = Question(
            session_id=question.session_id,
            question=decision.question,
            category=decision.category,
            priority=decision.priority,
            reason=decision.reason,
            status=QuestionStatus.PENDING,
            rank=None,
            parent_question_id=question.top_level_question_id,
        )
        db.add(next_question)
        db.commit()

    return AnswerResponse(
        type="follow_up" if decision.follow_up_required else "no_follow_up",
        follow_up_required=decision.follow_up_required,
        reason=decision.reason,
        question=_question_to_response(next_question) if next_question else None,
    )


@router.post("/sessions/{session_id}/complete", response_model=AssessmentReportSchema)
def post_complete_session(session_id: str, db: DBSession = Depends(get_db)):
    """Generate the Phase 3 due-diligence report for a completed session.

    Idempotent: a second call for the same session returns the existing
    report rather than generating a new one.
    """
    try:
        return complete_session(db, session_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IncompleteSessionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ReportGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/report", response_model=AssessmentReportSchema)
def get_session_report(session_id: str, db: DBSession = Depends(get_db)):
    """Return the stored Phase 3 report for a session. Does not generate one
    — use POST /sessions/{session_id}/complete for that.
    """
    try:
        return get_report(db, session_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def create_app() -> FastAPI:
    init_db()
    app = FastAPI(title="RAG Due-Diligence Questioning — Phase 2")
    app.include_router(router)
    return app
