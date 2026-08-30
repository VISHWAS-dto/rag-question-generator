"""The interview state machine: create a session (seed the top-N questions),
accept answers, drive the follow-up / next-question flow, complete the
session, and generate + store the report.

Ported from the original app/session_manager.py. Behaviour preserved,
including the concurrency guards (UNIQUE constraint on answers, idempotent
report generation, follow-up depth cap). Differences:
  * engines + clients + settings are injected, not imported as singletons;
  * errors are the typed set in app/orchestration/errors.py;
  * an idempotency key on create_session prevents duplicate interviews from a
    retried request;
  * empty-retrieval / dependency failures surface as typed errors, not 500s.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.config import AppSettings
from shared.logging import get_logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DBSession

from app.clients.llm import LLMClient
from app.clients.rag import RAGClient
from app.domain.schemas import AssessmentReportSchema, FollowupDecision
from app.engines.analyzer import InterviewTurn
from app.engines.followup import (
    FollowupContext,
    FollowupDecisionError,
    PriorTurn,
    decide_followup,
    retrieve_followup_context,
)
from app.engines.question_generator import QuestionGenerationError, generate_top_questions
from app.orchestration.errors import (
    ConflictError,
    DependencyUnavailableError,
    IncompleteSessionError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)
from app.orchestration.report_service import ReportGenerationError, build_report
from app.persistence.models import (
    Answer,
    AssessmentReport,
    AssessmentSession,
    Question,
    QuestionStatus,
    SessionStatus,
)

log = get_logger("app.orchestration.interview")


@dataclass
class AnswerResult:
    decision: FollowupDecision
    next_question: Question | None
    session_completed: bool


class InterviewService:
    def __init__(self, *, llm: LLMClient, rag: RAGClient, settings: AppSettings) -> None:
        self._llm = llm
        self._rag = rag
        self._settings = settings

    # --------------------------------------------------------------------- #
    # Create
    # --------------------------------------------------------------------- #

    def create_session(
        self,
        db: DBSession,
        *,
        company_id: str,
        startup_info: str,
        startup_stage: str | None = None,
        idempotency_key: str | None = None,
    ) -> AssessmentSession:
        company_id = (company_id or "").strip()
        startup_info = (startup_info or "").strip()
        if not company_id:
            raise ValidationError("company_id must not be empty")
        if not startup_info:
            raise ValidationError("startup_info must not be empty")
        if len(company_id) > self._settings.max_company_id_chars:
            raise ValidationError(
                f"company_id exceeds {self._settings.max_company_id_chars} characters"
            )
        if len(startup_info) > self._settings.max_startup_info_chars:
            raise ValidationError(
                f"startup_info exceeds {self._settings.max_startup_info_chars} characters"
            )

        if idempotency_key:
            existing = (
                db.query(AssessmentSession)
                .filter_by(idempotency_key=idempotency_key)
                .one_or_none()
            )
            if existing is not None:
                log.info("create_session_idempotent_hit", session_id=existing.session_id)
                return existing

        session = AssessmentSession(
            company_id=company_id,
            startup_info=startup_info,
            startup_stage=(startup_stage or None),
            status=SessionStatus.IN_PROGRESS,
            idempotency_key=idempotency_key,
        )
        db.add(session)
        db.flush()

        try:
            top_questions = generate_top_questions(
                llm=self._llm,
                rag=self._rag,
                startup_info=startup_info,
                startup_stage=startup_stage,
                num_questions=self._settings.num_questions,
                top_k=self._settings.rag_retrieval_top_k,
                collection=self._settings.rag_collection,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
                max_repair_attempts=self._settings.llm_max_repair_attempts,
            )
        except QuestionGenerationError as exc:
            db.rollback()
            raise UpstreamError(f"Question generation failed while creating the session: {exc}") from exc

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

        try:
            db.commit()
        except IntegrityError:
            # A concurrent request with the same idempotency_key won the race.
            db.rollback()
            if idempotency_key:
                existing = (
                    db.query(AssessmentSession)
                    .filter_by(idempotency_key=idempotency_key)
                    .one_or_none()
                )
                if existing is not None:
                    return existing
            raise
        db.refresh(session)
        return session

    # --------------------------------------------------------------------- #
    # Read
    # --------------------------------------------------------------------- #

    def get_session(self, db: DBSession, session_id: str) -> AssessmentSession:
        session = db.get(AssessmentSession, session_id)
        if session is None:
            raise NotFoundError(f"Session {session_id} not found")
        return session

    def get_question(self, db: DBSession, question_id: str) -> Question:
        question = db.get(Question, question_id)
        if question is None:
            raise NotFoundError(f"Question {question_id} not found")
        return question

    def list_questions(self, db: DBSession, session_id: str) -> list[Question]:
        return list(self.get_session(db, session_id).questions)

    def get_current_question(self, db: DBSession, session_id: str) -> Question | None:
        session = self.get_session(db, session_id)
        top_level = sorted(
            (q for q in session.questions if not q.is_followup), key=lambda q: q.rank or 0
        )
        for top in top_level:
            thread = [top, *sorted(top.follow_ups, key=lambda q: q.created_at)]
            pending = [q for q in thread if q.status == QuestionStatus.PENDING]
            if pending:
                return pending[0]
        return None

    # --------------------------------------------------------------------- #
    # Answer flow
    # --------------------------------------------------------------------- #

    def submit_answer(
        self, db: DBSession, *, question_id: str, answer_text: str
    ) -> AnswerResult:
        answer_text = (answer_text or "").strip()
        if not answer_text:
            raise ValidationError("answer must not be empty")
        if len(answer_text) > self._settings.max_answer_chars:
            raise ValidationError(
                f"answer exceeds {self._settings.max_answer_chars} characters"
            )

        question = self.get_question(db, question_id)
        if question.status == QuestionStatus.ANSWERED:
            raise ConflictError(f"Question {question_id} has already been answered")

        session = question.session

        db.add(
            Answer(
                question_id=question.question_id,
                session_id=session.session_id,
                answer=answer_text,
            )
        )
        question.status = QuestionStatus.ANSWERED
        try:
            db.flush()
        except IntegrityError as exc:
            # Race: two concurrent submits both read PENDING. The answers
            # UNIQUE(question_id) constraint is the real guard.
            db.rollback()
            raise ConflictError(
                f"Question {question_id} has already been answered"
            ) from exc
        db.refresh(session)

        # --- Build follow-up context (pure), retrieve grounding, decide ---
        top_level = self._top_level(session, question)
        previous_turns = [
            PriorTurn(question=q.question, answer=q.answer.answer)
            for q in session.questions
            if q.question_id != question.question_id
            and q.status == QuestionStatus.ANSWERED
            and q.answer is not None
        ]
        existing_followups = [
            q.question
            for q in top_level.follow_ups
            if q.question_id != question.question_id
        ]

        try:
            rag_context = retrieve_followup_context(
                self._rag,
                current_question=question.question,
                current_answer=answer_text,
                top_k=self._settings.rag_followup_top_k,
                collection=self._settings.rag_collection,
            )
        except FollowupDecisionError as exc:
            db.rollback()
            raise DependencyUnavailableError(
                f"Could not retrieve context for the follow-up decision: {exc}"
            ) from exc

        ctx = FollowupContext(
            startup_info=session.startup_info,
            startup_stage=session.startup_stage,
            rag_context=rag_context,
            top_level_question=top_level.question,
            current_question=question.question,
            current_answer=answer_text,
            previous_turns=previous_turns,
            existing_followup_questions=existing_followups,
        )

        try:
            decision = decide_followup(
                llm=self._llm,
                ctx=ctx,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
                max_repair_attempts=self._settings.llm_max_repair_attempts,
            )
        except FollowupDecisionError as exc:
            db.rollback()
            raise UpstreamError(f"Follow-up decision failed: {exc}") from exc

        # --- Depth cap ---
        top_level = self.get_question(db, question.top_level_question_id)
        if (
            decision.follow_up_required
            and len(top_level.follow_ups) >= self._settings.max_followups_per_question
        ):
            decision = FollowupDecision(
                follow_up_required=False,
                question=None,
                category=None,
                priority=None,
                reason=(
                    f"Suppressed: reached the maximum of "
                    f"{self._settings.max_followups_per_question} follow-ups for this "
                    f"question. Moving on to the next Top-10 question; original reason "
                    f"for a further follow-up: {decision.reason}"
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
            next_question = self.get_current_question(db, session.session_id)
            if next_question is None:
                session.status = SessionStatus.COMPLETED
                db.commit()

        return AnswerResult(
            decision=decision,
            next_question=next_question,
            session_completed=next_question is None,
        )

    # --------------------------------------------------------------------- #
    # Completion + report
    # --------------------------------------------------------------------- #

    def is_session_complete(self, db: DBSession, session_id: str) -> bool:
        return self.get_current_question(db, session_id) is None

    def complete_session(self, db: DBSession, session_id: str) -> AssessmentReportSchema:
        session = self.get_session(db, session_id)

        existing = (
            db.query(AssessmentReport).filter_by(session_id=session_id).one_or_none()
        )
        if existing is not None:
            return _report_row_to_schema(existing)

        if not self.is_session_complete(db, session_id):
            raise IncompleteSessionError(
                f"Session {session_id} is not complete: an unanswered question or pending "
                "follow-up remains. All questions must be answered before a report can be "
                "generated."
            )

        turns = [
            InterviewTurn(
                question=q.question,
                category=q.category,
                is_followup=q.is_followup,
                answer=q.answer.answer,
            )
            for q in session.questions
            if q.answer is not None
        ]

        try:
            report_schema = build_report(
                llm=self._llm,
                rag=self._rag,
                settings=self._settings,
                session_id=session_id,
                startup_info=session.startup_info,
                startup_stage=session.startup_stage,
                turns=turns,
            )
        except ReportGenerationError as exc:
            raise UpstreamError(str(exc)) from exc

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
                (cat.value if hasattr(cat, "value") else cat): score
                for cat, score in report_schema.category_scores.items()
            },
            recommendations=[r.model_dump(mode="json") for r in report_schema.recommendations],
        )
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = (
                db.query(AssessmentReport).filter_by(session_id=session_id).one_or_none()
            )
            if existing is not None:
                return _report_row_to_schema(existing)
            raise
        db.refresh(row)
        return _report_row_to_schema(row)

    def get_report(self, db: DBSession, session_id: str) -> AssessmentReportSchema:
        self.get_session(db, session_id)
        row = db.query(AssessmentReport).filter_by(session_id=session_id).one_or_none()
        if row is None:
            raise NotFoundError(f"No report exists yet for session {session_id}")
        return _report_row_to_schema(row)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _top_level(session: AssessmentSession, question: Question) -> Question:
        if not question.is_followup:
            return question
        for q in session.questions:
            if q.question_id == question.parent_question_id:
                return q
        raise ValidationError(
            f"Parent question {question.parent_question_id} not found in session"
        )


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
