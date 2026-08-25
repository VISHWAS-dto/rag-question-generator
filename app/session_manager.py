"""Phase 2 orchestration: create sessions, seed the top-10 questions, accept
answers, and drive the interactive follow-up / next-question flow.

Reuses Phase 1's generate_top_questions for seeding and the RAG retriever
via context_builder for follow-up grounding — no pipeline logic is
duplicated here.
"""

from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DBSession

from app.config import MAX_FOLLOWUPS_PER_QUESTION
from app.models import (
    Answer,
    AssessmentReport,
    AssessmentSession,
    Question,
    QuestionStatus,
    SessionStatus,
)
from app.question_engine.context_builder import build_followup_context
from app.question_engine.followup import FollowupDecision, SupportsInvoke, decide_followup
from app.question_engine.generator import generate_top_questions
from app.report_engine.report_generator import generate_report
from app.report_engine.schemas import AssessmentReportSchema


class NotFoundError(LookupError):
    pass


class ValidationError(ValueError):
    pass


class IncompleteSessionError(ValueError):
    """Raised when report generation is requested for a session that still
    has an unanswered Top-10 question or an outstanding follow-up.
    """


@dataclass
class AnswerResult:
    decision: FollowupDecision
    next_question: Question | None
    session_completed: bool


def create_session(
    db: DBSession, company_id: str, startup_info: str, startup_stage: str | None = None
) -> AssessmentSession:
    """Create a session and seed it with the Phase 1 top-10 questions.

    The top-10 questions themselves are never modified after this point —
    only follow-ups are added, linked to the relevant top-10 question.
    """
    if not startup_info or not startup_info.strip():
        raise ValidationError("startup_info must not be empty")

    session = AssessmentSession(
        company_id=company_id,
        startup_info=startup_info.strip(),
        startup_stage=startup_stage,
        status=SessionStatus.IN_PROGRESS,
    )
    db.add(session)
    db.flush()

    top_questions = generate_top_questions(startup_info, startup_stage=startup_stage)
    for rank, q in enumerate(top_questions, start=1):
        db.add(
            Question(
                session_id=session.session_id,
                question=q.question,
                category=q.category,
                priority=q.priority,
                reason=q.reason,
                status=QuestionStatus.PENDING,
                rank=rank,
            )
        )

    db.commit()
    db.refresh(session)
    return session


def get_session(db: DBSession, session_id: str) -> AssessmentSession:
    session = db.get(AssessmentSession, session_id)
    if session is None:
        raise NotFoundError(f"Session {session_id} not found")
    return session


def get_question(db: DBSession, question_id: str) -> Question:
    question = db.get(Question, question_id)
    if question is None:
        raise NotFoundError(f"Question {question_id} not found")
    return question


def list_questions(db: DBSession, session_id: str) -> list[Question]:
    session = get_session(db, session_id)
    return list(session.questions)


def get_current_question(db: DBSession, session_id: str) -> Question | None:
    """The question the user should answer next.

    Walks the Top-10 questions in rank order. For the first one whose thread
    (itself plus all its follow-ups) still has a pending question, returns
    the oldest pending question in that thread — a follow-up must always be
    completed before the next Top-10 question is offered.

    This must NOT be "oldest pending question by created_at" across the
    whole session: all 10 Top-10 questions are inserted in a single batch at
    session creation (same/near-identical timestamp), while a follow-up is
    always created later. Picking by raw created_at would therefore prefer
    an unanswered Top-10 question over a pending follow-up on an earlier
    Top-10 question, letting the interview skip ahead before the follow-up
    is resolved.
    """
    session = get_session(db, session_id)
    top_level = sorted(
        (q for q in session.questions if not q.is_followup), key=lambda q: q.rank or 0
    )

    for top in top_level:
        thread = [top, *sorted(top.follow_ups, key=lambda q: q.created_at)]
        pending_in_thread = [q for q in thread if q.status == QuestionStatus.PENDING]
        if pending_in_thread:
            return pending_in_thread[0]

    return None


def submit_answer(
    db: DBSession, question_id: str, answer_text: str, llm: SupportsInvoke | None = None
) -> AnswerResult:
    """Full answer flow: validate, store, retrieve RAG context, decide on a
    follow-up, and return either the new follow-up or the next unanswered
    top-10 question.
    """
    if not answer_text or not answer_text.strip():
        raise ValidationError("answer must not be empty")

    question = get_question(db, question_id)
    if question.status == QuestionStatus.ANSWERED:
        raise ValidationError(f"Question {question_id} has already been answered")

    session = question.session

    answer = Answer(
        question_id=question.question_id,
        session_id=session.session_id,
        answer=answer_text.strip(),
    )
    db.add(answer)
    question.status = QuestionStatus.ANSWERED
    try:
        db.flush()
    except IntegrityError as exc:
        # The status check above has a race window: two concurrent requests
        # for the same question can both read PENDING before either commits.
        # The `answers.question_id` UNIQUE constraint is the actual guard;
        # translate its violation into the same ValidationError a
        # non-concurrent duplicate submission raises, instead of letting a
        # raw IntegrityError surface as an unhandled 500.
        db.rollback()
        raise ValidationError(f"Question {question_id} has already been answered") from exc
    db.refresh(session)

    ctx = build_followup_context(session, question, answer_text.strip())
    decision = decide_followup(ctx, llm=llm)

    top_level = get_question(db, question.top_level_question_id)
    existing_followup_count = len(top_level.follow_ups)
    if decision.follow_up_required and existing_followup_count >= MAX_FOLLOWUPS_PER_QUESTION:
        # Defensive cap: without this, a chain of vague/evasive answers can
        # keep opening new, individually-reasonable follow-up threads
        # indefinitely (observed live: 11+ deep, never converging), which
        # would prevent the interview from ever completing. Force a move to
        # the next Top-10 question instead of asking another follow-up.
        decision = FollowupDecision(
            follow_up_required=False,
            question=None,
            category=None,
            priority=None,
            reason=(
                f"Suppressed: reached the maximum of {MAX_FOLLOWUPS_PER_QUESTION} "
                "follow-ups for this question. Moving on to the next Top-10 question; "
                f"original reason for a further follow-up: {decision.reason}"
            ),
        )

    next_question: Question | None = None

    if decision.follow_up_required and decision.question:
        next_question = Question(
            session_id=session.session_id,
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
    else:
        db.commit()
        db.refresh(session)
        next_question = get_current_question(db, session.session_id)
        if next_question is None:
            session.status = SessionStatus.COMPLETED
            db.commit()

    session_completed = next_question is None
    return AnswerResult(
        decision=decision, next_question=next_question, session_completed=session_completed
    )


# --- Phase 3: session completion + report generation ---


def is_session_complete(db: DBSession, session_id: str) -> bool:
    """A session is eligible for a final report only when every Top-10
    question has been answered and there is no pending follow-up anywhere in
    the session — i.e. get_current_question returns nothing left to ask.
    """
    return get_current_question(db, session_id) is None


def complete_session(
    db: DBSession, session_id: str, llm: SupportsInvoke | None = None
) -> AssessmentReportSchema:
    """Generate (or, if one already exists, return) the final Phase 3 report
    for a completed session.

    Idempotent: calling this twice for the same session does not generate a
    second report — the existing stored report is returned as-is, without
    re-running analysis.

    Raises IncompleteSessionError if the interview isn't actually finished
    (an unanswered Top-10 question or a pending follow-up remains), and
    ReportGenerationError if RAG retrieval, the LLM call, or report
    validation fails.
    """
    session = get_session(db, session_id)

    existing = db.query(AssessmentReport).filter_by(session_id=session_id).one_or_none()
    if existing is not None:
        return _report_row_to_schema(existing)

    if not is_session_complete(db, session_id):
        raise IncompleteSessionError(
            f"Session {session_id} is not complete: an unanswered Top-10 question or "
            "pending follow-up remains. All questions must be answered before a report "
            "can be generated."
        )

    report_schema = generate_report(session, llm=llm)

    session.status = SessionStatus.COMPLETED
    row = AssessmentReport(
        report_id=report_schema.report_id,
        session_id=session_id,
        overall_score=report_schema.overall_score,
        risk_level=report_schema.risk_level.value,
        executive_summary=report_schema.executive_summary,
        strengths=[s.model_dump(mode="json") for s in report_schema.strengths],
        risks=[r.model_dump(mode="json") for r in report_schema.risks],
        information_gaps=[g.model_dump(mode="json") for g in report_schema.information_gaps],
        contradictions=[c.model_dump(mode="json") for c in report_schema.contradictions],
        category_scores={
            cat.value if hasattr(cat, "value") else cat: score
            for cat, score in report_schema.category_scores.items()
        },
        recommendations=[r.model_dump(mode="json") for r in report_schema.recommendations],
    )
    db.add(row)

    # Guard against a race: a concurrent request may have inserted a report
    # for this session between the earlier existence check and this commit
    # (session_id is UNIQUE on assessment_reports). Fall back to returning
    # that report rather than erroring, keeping this endpoint idempotent.
    try:
        db.commit()
    except Exception:
        db.rollback()
        existing = db.query(AssessmentReport).filter_by(session_id=session_id).one_or_none()
        if existing is not None:
            return _report_row_to_schema(existing)
        raise

    db.refresh(row)
    return _report_row_to_schema(row)


def get_report(db: DBSession, session_id: str) -> AssessmentReportSchema:
    """Fetch the stored report for a session. Raises NotFoundError if the
    session doesn't exist, or if it exists but no report has been generated
    yet (report generation is never implicitly triggered by a GET).
    """
    get_session(db, session_id)  # raises NotFoundError if the session itself is missing

    row = db.query(AssessmentReport).filter_by(session_id=session_id).one_or_none()
    if row is None:
        raise NotFoundError(f"No report exists yet for session {session_id}")
    return _report_row_to_schema(row)


def _report_row_to_schema(row: AssessmentReport) -> AssessmentReportSchema:
    return AssessmentReportSchema(
        report_id=row.report_id,
        session_id=row.session_id,
        overall_score=row.overall_score,
        risk_level=row.risk_level,
        executive_summary=row.executive_summary,
        strengths=row.strengths,
        risks=row.risks,
        information_gaps=row.information_gaps,
        contradictions=row.contradictions,
        category_scores=row.category_scores,
        recommendations=row.recommendations,
        created_at=row.created_at.isoformat(),
    )
