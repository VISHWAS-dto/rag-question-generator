"""Interview state machine (app/orchestration/interview_service.py).

Ports tests/test_phase2.py + the report-flow parts of tests/test_phase3.py to
the injected-dependency design: `FakeLLMClient` scripts the model responses,
`FakeRAGClient` serves canned chunks, DB is in-memory SQLite.
"""

from __future__ import annotations

import pytest
from app.clients.llm import FakeLLMClient
from app.clients.rag import FakeRAGClient, RAGUnavailableError
from app.orchestration.errors import (
    ConflictError,
    DependencyUnavailableError,
    IncompleteSessionError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)
from app.orchestration.interview_service import InterviewService
from app.persistence.models import Answer, AssessmentReport, QuestionStatus

from tests.conftest import (
    analysis_payload,
    followup_payload,
    make_completed_session,
    make_session_with_questions,
    no_followup_payload,
    top_questions_payload,
)

pytestmark = pytest.mark.unit


def _svc(llm: FakeLLMClient, rag: FakeRAGClient, settings) -> InterviewService:
    return InterviewService(llm=llm, rag=rag, settings=settings)


# --------------------------------------------------------------------------- #
# create_session
# --------------------------------------------------------------------------- #


def test_create_session_seeds_questions(db, fake_rag, settings):
    llm = FakeLLMClient(responses=[top_questions_payload(3)])
    svc = _svc(llm, fake_rag, settings)
    session = svc.create_session(
        db, company_id="acme", startup_info="B2B SaaS, 15 staff.", startup_stage="Seed"
    )
    questions = svc.list_questions(db, session.session_id)
    assert len(questions) == 3
    assert all(q.rank is not None for q in questions)


@pytest.mark.parametrize(
    "company_id,startup_info,err",
    [
        ("", "info", ValidationError),
        ("acme", "", ValidationError),
        ("acme", "   ", ValidationError),
    ],
)
def test_create_session_rejects_empty_fields(db, fake_rag, settings, company_id, startup_info, err):
    svc = _svc(FakeLLMClient(responses=[top_questions_payload()]), fake_rag, settings)
    with pytest.raises(err):
        svc.create_session(db, company_id=company_id, startup_info=startup_info)


def test_create_session_rejects_oversized_startup_info(db, fake_rag, settings):
    svc = _svc(FakeLLMClient(responses=[top_questions_payload()]), fake_rag, settings)
    with pytest.raises(ValidationError):
        svc.create_session(
            db, company_id="acme", startup_info="x" * (settings.max_startup_info_chars + 1)
        )


def test_create_session_idempotency_key_returns_same_session(db, fake_rag, settings):
    llm = FakeLLMClient(responses=[top_questions_payload(3), top_questions_payload(3)])
    svc = _svc(llm, fake_rag, settings)
    s1 = svc.create_session(
        db, company_id="acme", startup_info="info", idempotency_key="abc-123"
    )
    s2 = svc.create_session(
        db, company_id="acme", startup_info="info", idempotency_key="abc-123"
    )
    assert s1.session_id == s2.session_id
    assert len(llm.prompts_seen) == 1  # second call did not hit the LLM


def test_create_session_llm_failure_is_upstream_error(db, fake_rag, settings):
    # No scripted responses -> FakeLLMClient raises -> QuestionGenerationError -> UpstreamError
    svc = _svc(FakeLLMClient(responses=[]), fake_rag, settings)
    with pytest.raises(UpstreamError):
        svc.create_session(db, company_id="acme", startup_info="info")


def test_create_session_empty_retrieval_is_upstream_error(db, settings):
    rag = FakeRAGClient(chunks=[])
    svc = _svc(FakeLLMClient(responses=[top_questions_payload()]), rag, settings)
    with pytest.raises(UpstreamError, match="knowledge base"):
        svc.create_session(db, company_id="acme", startup_info="info")


def test_create_session_rag_down_is_upstream_error(db, settings):
    rag = FakeRAGClient(raise_error=RAGUnavailableError("rag offline"))
    svc = _svc(FakeLLMClient(responses=[top_questions_payload()]), rag, settings)
    with pytest.raises(UpstreamError):
        svc.create_session(db, company_id="acme", startup_info="info")


# --------------------------------------------------------------------------- #
# submit_answer
# --------------------------------------------------------------------------- #


def test_answer_stored_and_linked(db, fake_rag, settings):
    session = make_session_with_questions(db)
    q1 = session.questions[0]
    svc = _svc(FakeLLMClient(responses=[no_followup_payload()]), fake_rag, settings)
    result = svc.submit_answer(db, question_id=q1.question_id, answer_text="70%")
    db.refresh(q1)
    assert q1.status == QuestionStatus.ANSWERED
    assert q1.answer.answer == "70%"
    assert result.decision.follow_up_required is False


@pytest.mark.parametrize("answer", ["", "   "])
def test_empty_answer_rejected(db, fake_rag, settings, answer):
    session = make_session_with_questions(db)
    svc = _svc(FakeLLMClient(responses=[no_followup_payload()]), fake_rag, settings)
    with pytest.raises(ValidationError):
        svc.submit_answer(db, question_id=session.questions[0].question_id, answer_text=answer)


def test_oversized_answer_rejected(db, fake_rag, settings):
    session = make_session_with_questions(db)
    svc = _svc(FakeLLMClient(responses=[no_followup_payload()]), fake_rag, settings)
    with pytest.raises(ValidationError):
        svc.submit_answer(
            db,
            question_id=session.questions[0].question_id,
            answer_text="x" * (settings.max_answer_chars + 1),
        )


def test_followup_generated_when_answer_reveals_issue(db, fake_rag, settings):
    session = make_session_with_questions(db)
    q1 = session.questions[0]
    svc = _svc(
        FakeLLMClient(
            responses=[followup_payload("What are you doing to reduce that dependency?")]
        ),
        fake_rag,
        settings,
    )
    result = svc.submit_answer(db, question_id=q1.question_id, answer_text="70%")
    assert result.decision.follow_up_required is True
    assert result.next_question.is_followup is True
    assert result.next_question.parent_question_id == q1.question_id


def test_no_followup_advances_to_next_question(db, fake_rag, settings):
    session = make_session_with_questions(db)
    q1, q2 = session.questions[0], session.questions[1]
    svc = _svc(FakeLLMClient(responses=[no_followup_payload()]), fake_rag, settings)
    result = svc.submit_answer(db, question_id=q1.question_id, answer_text="70%")
    assert result.next_question.question_id == q2.question_id
    assert result.next_question.is_followup is False


def test_duplicate_followup_is_suppressed(db, fake_rag, settings):
    session = make_session_with_questions(db)
    q1 = session.questions[0]
    # LLM proposes a follow-up that duplicates the current question itself.
    svc = _svc(
        FakeLLMClient(responses=[followup_payload(q1.question)]), fake_rag, settings
    )
    result = svc.submit_answer(db, question_id=q1.question_id, answer_text="70%")
    assert result.decision.follow_up_required is False


def test_followup_depth_cap_forces_move_on(db, fake_rag, settings):
    session = make_session_with_questions(db)
    q1, q2 = session.questions[0], session.questions[1]
    n = settings.max_followups_per_question
    # Always wants another follow-up, distinct each time so only the cap stops it.
    responses = [followup_payload(f"Follow-up number {i}?") for i in range(1, n + 5)]
    svc = _svc(FakeLLMClient(responses=responses), fake_rag, settings)

    result = svc.submit_answer(db, question_id=q1.question_id, answer_text="70%")
    seen = 0
    while result.decision.follow_up_required:
        seen += 1
        assert seen <= n
        result = svc.submit_answer(
            db, question_id=result.next_question.question_id, answer_text="still vague"
        )
    assert seen == n
    assert result.next_question.question_id == q2.question_id
    assert "maximum" in result.decision.reason.lower()


def test_already_answered_question_conflicts(db, fake_rag, settings):
    session = make_session_with_questions(db)
    q1 = session.questions[0]
    svc = _svc(
        FakeLLMClient(responses=[no_followup_payload(), no_followup_payload()]),
        fake_rag,
        settings,
    )
    svc.submit_answer(db, question_id=q1.question_id, answer_text="70%")
    with pytest.raises(ConflictError):
        svc.submit_answer(db, question_id=q1.question_id, answer_text="again")


def test_concurrent_duplicate_answer_race_becomes_conflict(db, fake_rag, settings):
    session = make_session_with_questions(db)
    q1 = session.questions[0]
    svc = _svc(FakeLLMClient(responses=[no_followup_payload()]), fake_rag, settings)
    svc.submit_answer(db, question_id=q1.question_id, answer_text="70%")

    # Simulate the race: a second Answer row inserted directly -> UNIQUE violation.
    from sqlalchemy.exc import IntegrityError

    db.add(Answer(question_id=q1.question_id, session_id=session.session_id, answer="dup"))
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()


def test_unknown_question_is_not_found(db, fake_rag, settings):
    svc = _svc(FakeLLMClient(responses=[no_followup_payload()]), fake_rag, settings)
    with pytest.raises(NotFoundError):
        svc.submit_answer(db, question_id="does-not-exist", answer_text="hi")


def test_rag_down_during_followup_is_dependency_unavailable(db, settings):
    session = make_session_with_questions(db)
    q1 = session.questions[0]
    rag = FakeRAGClient(raise_error=RAGUnavailableError("rag offline"))
    svc = _svc(FakeLLMClient(responses=[no_followup_payload()]), rag, settings)
    with pytest.raises(DependencyUnavailableError):
        svc.submit_answer(db, question_id=q1.question_id, answer_text="70%")
    # The answer write was rolled back, so the question is still answerable.
    db.expire_all()
    assert db.get(type(q1), q1.question_id).status == QuestionStatus.PENDING


def test_malformed_followup_output_repairs_then_succeeds(db, fake_rag, settings):
    session = make_session_with_questions(db)
    q1 = session.questions[0]
    svc = _svc(
        FakeLLMClient(
            responses=[
                "sorry, here it is: {follow_up_required: true",  # malformed
                followup_payload("What is your enterprise revenue share?"),  # repaired
            ]
        ),
        fake_rag,
        settings,
    )
    result = svc.submit_answer(db, question_id=q1.question_id, answer_text="70%")
    assert result.decision.follow_up_required is True


# --------------------------------------------------------------------------- #
# current-question sequencing
# --------------------------------------------------------------------------- #


def test_pending_followup_precedes_next_top_level(db, fake_rag, settings):
    session = make_session_with_questions(db)
    q1, q2 = session.questions[0], session.questions[1]
    svc = _svc(
        FakeLLMClient(responses=[followup_payload("Reduce dependency how?")]),
        fake_rag,
        settings,
    )
    result = svc.submit_answer(db, question_id=q1.question_id, answer_text="70%")
    followup = result.next_question
    db.expire_all()
    current = svc.get_current_question(db, session.session_id)
    assert current.question_id == followup.question_id
    assert current.question_id != q2.question_id


# --------------------------------------------------------------------------- #
# completion + report
# --------------------------------------------------------------------------- #


def test_report_generated_for_completed_session(db, fake_rag, settings):
    session = make_completed_session(db)
    svc = _svc(FakeLLMClient(responses=[analysis_payload()]), fake_rag, settings)
    report = svc.complete_session(db, session.session_id)
    assert report.session_id == session.session_id
    assert report.executive_summary
    assert len(report.category_scores) == 10


def test_incomplete_session_rejected(db, fake_rag, settings):
    session = make_session_with_questions(db)  # nothing answered
    svc = _svc(FakeLLMClient(responses=[analysis_payload()]), fake_rag, settings)
    with pytest.raises(IncompleteSessionError):
        svc.complete_session(db, session.session_id)


def test_report_is_idempotent(db, fake_rag, settings):
    session = make_completed_session(db)
    svc = _svc(
        FakeLLMClient(responses=[analysis_payload(), analysis_payload()]),
        fake_rag,
        settings,
    )
    r1 = svc.complete_session(db, session.session_id)
    r2 = svc.complete_session(db, session.session_id)
    assert r1.report_id == r2.report_id
    assert db.query(AssessmentReport).filter_by(session_id=session.session_id).count() == 1


def test_report_malformed_output_repairs(db, fake_rag, settings):
    session = make_completed_session(db)
    svc = _svc(
        FakeLLMClient(
            responses=['{"executive_summary": "truncated mid', analysis_payload()]
        ),
        fake_rag,
        settings,
    )
    report = svc.complete_session(db, session.session_id)
    assert report.executive_summary


def test_severe_contradiction_escalated_in_report(db, fake_rag, settings):
    session = make_completed_session(
        db,
        answer="Our annual revenue is 5 crore INR.",
        second_answer="Actually, our annual revenue is 3 crore INR.",
    )
    payload = analysis_payload(
        contradictions=[
            {
                "topic": "Annual revenue",
                "earlier_claim": "Our annual revenue is 5 crore INR.",
                "later_claim": "Actually, our annual revenue is 3 crore INR.",
                "explanation": "Revenue figures inconsistent.",
                "severity": "HIGH",
            }
        ]
    )
    svc = _svc(FakeLLMClient(responses=[payload]), fake_rag, settings)
    report = svc.complete_session(db, session.session_id)
    assert len(report.contradictions) == 1
    assert any("revenue" in r.title.lower() for r in report.risks)


def test_get_report_before_generation_is_not_found(db, fake_rag, settings):
    session = make_completed_session(db)
    svc = _svc(FakeLLMClient(responses=[analysis_payload()]), fake_rag, settings)
    with pytest.raises(NotFoundError):
        svc.get_report(db, session.session_id)


def test_report_llm_failure_is_upstream_error(db, fake_rag, settings):
    session = make_completed_session(db)
    svc = _svc(FakeLLMClient(responses=[]), fake_rag, settings)  # no responses -> raises
    with pytest.raises(UpstreamError):
        svc.complete_session(db, session.session_id)
