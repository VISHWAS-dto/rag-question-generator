"""Phase 2 orchestration: create sessions, seed the top-10 questions, accept
answers, and drive the interactive follow-up / next-question flow.

Reuses Phase 1's generate_top_questions for seeding and the RAG retriever
via context_builder for follow-up grounding — no pipeline logic is
duplicated here.
"""

from dataclasses import dataclass

from sqlalchemy.orm import Session as DBSession

from app.models import Answer, AssessmentSession, Question, QuestionStatus, SessionStatus
from app.question_engine.context_builder import build_followup_context
from app.question_engine.followup import FollowupDecision, SupportsInvoke, decide_followup
from app.question_engine.generator import generate_top_questions


class NotFoundError(LookupError):
    pass


class ValidationError(ValueError):
    pass


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

    top_questions = generate_top_questions(startup_info)
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
    """The question the user should answer next: the oldest PENDING question
    (a follow-up, if one is outstanding, otherwise the next top-10 question —
    follow-ups are created immediately after their parent's answer, so
    creation order already reflects the correct turn order).
    """
    session = get_session(db, session_id)
    pending = [q for q in session.questions if q.status == QuestionStatus.PENDING]
    if not pending:
        return None
    return min(pending, key=lambda q: q.created_at)


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
    db.flush()
    db.refresh(session)

    ctx = build_followup_context(session, question, answer_text.strip())
    decision = decide_followup(ctx, llm=llm)

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
