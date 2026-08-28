"""Phase 3 tests: report generation, scoring, risk detection, information
gaps, contradiction detection, idempotency, and the completion API.

Uses the same stub-LLM + in-memory SQLite approach as tests/test_phase2.py
for deterministic, offline results. RAG evidence retrieval hits the real
persisted ChromaDB store (scripts/build_index.py must have been run), same
as Phase 2's live-integration test does for its retrieval step; the LLM
analysis call itself is always stubbed here (no live-LLM test is included,
consistent with keeping this suite fast and deterministic).

Run with: python tests/test_phase3.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import (
    Answer,
    AssessmentReport,
    AssessmentSession,
    Base,
    Question,
    QuestionStatus,
    SessionStatus,
)
from app.report_engine.scorer import (
    compute_overall_score,
    determine_risk_level,
    score_categories,
)
from app.report_engine.schemas import (
    Category,
    CategoryAssessment,
    Contradiction,
    Risk,
    RiskLevel,
    Severity,
)
from app.session_manager import (
    IncompleteSessionError,
    NotFoundError,
    complete_session,
    get_report,
)

STARTUP_INFO = (
    "We're a B2B SaaS startup, 15 employees, ₹2 crore annual revenue. "
    "70% of our revenue comes from our top five customers."
)


# --- Test infrastructure ---


def make_test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


class StubResponse:
    def __init__(self, content: str):
        self.content = content


class StubLLM:
    """Returns a canned InterviewAnalysis JSON payload, like followup.py's
    StubLLM in test_phase2.py.
    """

    def __init__(self, analysis_dict):
        self._analysis_dict = analysis_dict

    def invoke(self, _prompt_value) -> StubResponse:
        return StubResponse(json.dumps(self._analysis_dict))


class RepairingStubLLM:
    """First `.invoke()` returns malformed JSON (exercising the generate ->
    parse -> repair LangGraph), then returns the canned analysis payload.
    """

    def __init__(self, analysis_dict, bad_responses=1):
        self._analysis_dict = analysis_dict
        self._bad_left = bad_responses
        self.calls = 0

    def invoke(self, _prompt_value) -> StubResponse:
        self.calls += 1
        if self._bad_left > 0:
            self._bad_left -= 1
            return StubResponse('{"executive_summary": "truncated mid-json')
        return StubResponse(json.dumps(self._analysis_dict))


def _all_category_assessments(
    assessment="Moderate", evidence_strength="Low", evidence_gaps=None
):
    return [
        {
            "category": c.value,
            "assessment": assessment,
            "rationale": "Limited detail provided in the interview.",
            "evidence_strength": evidence_strength,
            "evidence_gaps": evidence_gaps or ["quantitative detail"],
        }
        for c in Category
    ]


def base_analysis(**overrides) -> dict:
    analysis = {
        "executive_summary": "Early-stage B2B SaaS with meaningful customer concentration risk.",
        "strengths": [],
        "risks": [],
        "information_gaps": [],
        "contradictions": [],
        "category_assessments": _all_category_assessments(),
        "recommendations": [],
    }
    analysis.update(overrides)
    return analysis


def make_completed_session(db, extra_answer: str | None = None, second_answer: str | None = None):
    """A session with one Top-10 question, answered (and optionally a second
    question so contradiction tests have two data points) — fully complete,
    eligible for report generation.
    """
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
        reason="seeded for test",
        status=QuestionStatus.PENDING,
        rank=1,
    )
    db.add(q1)
    db.flush()
    db.add(
        Answer(
            question_id=q1.question_id,
            session_id=session.session_id,
            answer=extra_answer or "₹5 crore annual revenue.",
        )
    )
    q1.status = QuestionStatus.ANSWERED

    if second_answer is not None:
        q2 = Question(
            session_id=session.session_id,
            question="Can you confirm that revenue figure again?",
            category="Financial",
            priority="High",
            reason="seeded for test",
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


# --- Test 1: Report generation for a completed session ---


def test_report_generation() -> None:
    print("[1/10] Testing report generation for a completed session...")
    db = make_test_db()
    session = make_completed_session(db)

    stub = StubLLM(base_analysis())
    report = complete_session(db, session.session_id, llm=stub)

    assert report.session_id == session.session_id
    assert report.executive_summary
    assert len(report.category_scores) == len(Category)
    print(f"      OK - report generated, overall_score={report.overall_score}")


def test_report_repairs_malformed_llm_output() -> None:
    print("[1b/10] Testing report generation recovers from malformed LLM JSON...")
    db = make_test_db()
    session = make_completed_session(db)

    stub = RepairingStubLLM(base_analysis())
    report = complete_session(db, session.session_id, llm=stub)

    assert stub.calls == 2, "Graph should re-invoke once to repair the bad first response"
    assert report.session_id == session.session_id
    assert report.executive_summary
    assert len(report.category_scores) == len(Category)
    print("      OK - malformed first response repaired, valid report produced")


# --- Test 2: Incomplete session is rejected ---


def test_incomplete_session_rejected() -> None:
    print("[2/10] Testing incomplete session rejects report generation...")
    db = make_test_db()

    session = AssessmentSession(
        company_id="acme-co", startup_info=STARTUP_INFO, status=SessionStatus.IN_PROGRESS
    )
    db.add(session)
    db.flush()
    db.add(
        Question(
            session_id=session.session_id,
            question="What is your revenue?",
            category="Financial",
            priority="High",
            reason="seeded for test",
            status=QuestionStatus.PENDING,  # never answered
            rank=1,
        )
    )
    db.commit()
    db.refresh(session)

    try:
        complete_session(db, session.session_id, llm=StubLLM(base_analysis()))
        raise AssertionError("Expected IncompleteSessionError")
    except IncompleteSessionError:
        print("      OK - incomplete session correctly rejected")


# --- Test 3: Strength detection ---


def test_strength_detection() -> None:
    print("[3/10] Testing strength detection with supporting evidence...")
    db = make_test_db()
    session = make_completed_session(db)

    analysis = base_analysis(
        strengths=[
            {
                "title": "Strong revenue for stage",
                "description": "₹5 crore annual revenue at seed stage.",
                "category": "traction",
                "evidence": [{"source": "FOUNDER_ANSWER", "detail": "₹5 crore annual revenue."}],
                "confidence": "Medium",
            }
        ]
    )
    report = complete_session(db, session.session_id, llm=StubLLM(analysis))

    assert len(report.strengths) == 1
    assert report.strengths[0].evidence, "Strength must carry evidence"
    print(f"      OK - strength detected: '{report.strengths[0].title}'")


# --- Test 4: Risk detection (customer concentration) ---


def test_risk_detection() -> None:
    print("[4/10] Testing risk detection (customer concentration)...")
    db = make_test_db()
    session = make_completed_session(db, extra_answer="70% of revenue from our top 5 customers.")

    analysis = base_analysis(
        risks=[
            {
                "title": "Customer concentration",
                "description": "70% of revenue depends on 5 customers.",
                "category": "risk",
                "severity": "HIGH",
                "impact": "Loss of a top customer would materially harm revenue.",
                "evidence": [
                    {"source": "FOUNDER_ANSWER", "detail": "70% of revenue from our top 5 customers."}
                ],
                "confidence": "High",
            }
        ]
    )
    report = complete_session(db, session.session_id, llm=StubLLM(analysis))

    assert len(report.risks) == 1
    assert report.risks[0].severity == Severity.HIGH
    print(f"      OK - risk detected: '{report.risks[0].title}' (severity={report.risks[0].severity})")


# --- Test 5: Information gap detection ---


def test_information_gap_detection() -> None:
    print("[5/10] Testing information gap detection (missing retention data)...")
    db = make_test_db()
    session = make_completed_session(db)

    analysis = base_analysis(
        information_gaps=[
            {
                "topic": "Customer retention / churn",
                "why_it_matters": "No quantitative retention or churn data was provided.",
                "priority": "HIGH",
            }
        ]
    )
    report = complete_session(db, session.session_id, llm=StubLLM(analysis))

    assert len(report.information_gaps) == 1
    assert report.information_gaps[0].priority.value == "HIGH"
    print(f"      OK - information gap detected: '{report.information_gaps[0].topic}'")


# --- Test 6: Contradiction detection, aggregated across the interview ---


def test_contradiction_detection() -> None:
    print("[6/10] Testing contradiction detection (revenue 5cr vs 3cr)...")
    db = make_test_db()
    session = make_completed_session(
        db,
        extra_answer="Our annual revenue is ₹5 crore.",
        second_answer="Actually, our annual revenue is ₹3 crore.",
    )

    analysis = base_analysis(
        contradictions=[
            {
                "topic": "Annual revenue",
                "earlier_claim": "Our annual revenue is ₹5 crore.",
                "later_claim": "Actually, our annual revenue is ₹3 crore.",
                "explanation": "Revenue figures are inconsistent across the interview.",
                "severity": "HIGH",
            }
        ]
    )
    report = complete_session(db, session.session_id, llm=StubLLM(analysis))

    assert len(report.contradictions) == 1
    assert report.contradictions[0].severity == Severity.HIGH
    # A HIGH+ contradiction must also be escalated into risks (risk_analyzer.py).
    assert any("revenue" in r.title.lower() for r in report.risks), (
        "Severe contradiction should be escalated into the risks list"
    )
    print("      OK - contradiction detected and escalated to risks")


def test_non_genuine_contradiction_filtered() -> None:
    print("      Testing non-genuine contradiction is filtered out...")
    db = make_test_db()
    session = make_completed_session(db)

    analysis = base_analysis(
        contradictions=[
            {
                "topic": "Annual revenue",
                "earlier_claim": "₹5 crore annual revenue.",
                "later_claim": "₹5 crore annual revenue.",  # identical, not a real contradiction
                "explanation": "spurious",
                "severity": "HIGH",
            }
        ]
    )
    report = complete_session(db, session.session_id, llm=StubLLM(analysis))
    assert report.contradictions == [], "Identical claims must not be reported as a contradiction"
    print("      OK - non-genuine contradiction filtered by deterministic backstop")


# --- Test 7: Category scores always within [0, 10] ---


def test_category_scores_in_range() -> None:
    print("[7/10] Testing category scores are always within [0, 10]...")
    for assessment in ("Excellent", "Strong", "Moderate", "Weak", "Critical"):
        for strength in ("Low", "Medium", "High"):
            assessments = [
                CategoryAssessment(
                    category=Category.FINANCIALS,
                    assessment=assessment,
                    rationale="test",
                    evidence_strength=strength,
                    evidence_gaps=[],
                )
            ]
            scores = score_categories(assessments)
            score = scores[Category.FINANCIALS]
            assert 0 <= score <= 10, f"Score out of range: {score} for {assessment}/{strength}"
    print("      OK - all assessment/evidence-strength combinations produce scores in [0, 10]")


# --- Test 8: Overall score within [0, 10] and deterministic ---


def test_overall_score_bounds_and_determinism() -> None:
    print("[8/10] Testing overall score bounds and deterministic aggregation...")
    all_excellent = {cat: 10.0 for cat in Category}
    all_critical = {cat: 0.0 for cat in Category}
    mixed = {cat: 5.0 for cat in Category}

    assert compute_overall_score(all_excellent) == 10.0
    assert compute_overall_score(all_critical) == 0.0
    assert compute_overall_score(mixed) == 5.0

    # Determinism: same input -> same output, always.
    score_a = compute_overall_score(mixed)
    score_b = compute_overall_score(mixed)
    assert score_a == score_b

    for cat in Category:
        scores = dict(mixed)
        scores[cat] = 0.0
        overall = compute_overall_score(scores)
        assert 0 <= overall <= 10
    print("      OK - overall score always in [0, 10] and deterministic for fixed inputs")


# --- Test 9: Risk level thresholds, including critical-risk override ---


def test_risk_level_thresholds() -> None:
    print("[9/10] Testing risk level deterministic thresholds...")
    assert determine_risk_level(9.0, []) == RiskLevel.LOW
    assert determine_risk_level(8.0, []) == RiskLevel.LOW
    assert determine_risk_level(7.9, []) == RiskLevel.MEDIUM
    assert determine_risk_level(6.0, []) == RiskLevel.MEDIUM
    assert determine_risk_level(5.9, []) == RiskLevel.HIGH
    assert determine_risk_level(4.0, []) == RiskLevel.HIGH
    assert determine_risk_level(3.9, []) == RiskLevel.CRITICAL
    assert determine_risk_level(0.0, []) == RiskLevel.CRITICAL

    # A CRITICAL-severity risk overrides an otherwise-strong numeric score.
    critical_risk = Risk(
        title="Unresolved litigation",
        description="Active lawsuit threatens the company's core IP.",
        category=Category.RISK,
        severity=Severity.CRITICAL,
        impact="Could invalidate the company's core product.",
        evidence=[],
        confidence="High",
    )
    assert determine_risk_level(9.5, [critical_risk]) == RiskLevel.CRITICAL, (
        "A CRITICAL risk must override a high numeric score"
    )
    print("      OK - thresholds correct, and a CRITICAL risk overrides the numeric score")


# --- Test 10: Idempotency + API-level tests ---


def test_idempotency() -> None:
    print("[10/10] Testing idempotency: calling complete_session twice...")
    db = make_test_db()
    session = make_completed_session(db)

    stub = StubLLM(base_analysis())
    report1 = complete_session(db, session.session_id, llm=stub)
    report2 = complete_session(db, session.session_id, llm=stub)

    assert report1.report_id == report2.report_id, "A second call must not create a new report"
    count = db.query(AssessmentReport).filter_by(session_id=session.session_id).count()
    assert count == 1, f"Expected exactly one stored report, found {count}"
    print("      OK - idempotent: same report_id, exactly one row stored")


def test_get_report_not_found() -> None:
    print("      Testing GET report on a session with no report yet raises NotFoundError...")
    db = make_test_db()
    session = make_completed_session(db)
    try:
        get_report(db, session.session_id)
        raise AssertionError("Expected NotFoundError")
    except NotFoundError:
        print("      OK - NotFoundError raised when no report exists yet")


def test_get_report_after_complete() -> None:
    print("      Testing GET report after POST complete returns the stored report...")
    db = make_test_db()
    session = make_completed_session(db)
    stub = StubLLM(base_analysis())
    created = complete_session(db, session.session_id, llm=stub)
    fetched = get_report(db, session.session_id)
    assert fetched.report_id == created.report_id
    print("      OK - GET report matches the report created by POST complete")


# --- API-level tests using FastAPI's TestClient ---


def test_api_endpoints() -> None:
    print("      Testing API endpoints: POST /sessions/{id}/complete, GET /sessions/{id}/report...")
    from fastapi.testclient import TestClient

    from app.api import router
    from app.db import get_db
    from fastapi import FastAPI

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # single shared connection, so all sessions see the same in-memory DB
    )
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    db = TestSessionLocal()
    session = make_completed_session(db)
    db.close()

    # Incomplete-session case first, via a second session with an unanswered question.
    db2 = TestSessionLocal()
    incomplete_session = AssessmentSession(
        company_id="acme-co", startup_info=STARTUP_INFO, status=SessionStatus.IN_PROGRESS
    )
    db2.add(incomplete_session)
    db2.flush()
    db2.add(
        Question(
            session_id=incomplete_session.session_id,
            question="Unanswered?",
            category="Financial",
            priority="High",
            reason="seeded",
            status=QuestionStatus.PENDING,
            rank=1,
        )
    )
    db2.commit()
    db2.refresh(incomplete_session)
    db2.close()

    resp = client.post(f"/sessions/{incomplete_session.session_id}/complete")
    assert resp.status_code == 409, f"Expected 409 for incomplete session, got {resp.status_code}"

    # Report doesn't exist yet.
    resp = client.get(f"/sessions/{session.session_id}/report")
    assert resp.status_code == 404, f"Expected 404 before report exists, got {resp.status_code}"

    # Monkeypatch the LLM used by complete_session via session_manager's generate_report path:
    # simplest here is to directly patch app.report_engine.analyzer.get_llm.
    import app.report_engine.analyzer as analyzer_module

    stub_llm = StubLLM(base_analysis())
    analyzer_module._llm = stub_llm  # bypass require_nvidia_api_key + real ChatNVIDIA construction

    resp = client.post(f"/sessions/{session.session_id}/complete")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body1 = resp.json()
    assert body1["session_id"] == session.session_id

    resp2 = client.post(f"/sessions/{session.session_id}/complete")
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body1["report_id"] == body2["report_id"], "API-level idempotency failed"

    resp3 = client.get(f"/sessions/{session.session_id}/report")
    assert resp3.status_code == 200
    assert resp3.json()["report_id"] == body1["report_id"]

    analyzer_module._llm = None  # reset global for any subsequent test/run
    print("      OK - API endpoints: 409 incomplete, 404 no report, 200 create, idempotent, 200 GET")


def main() -> None:
    test_report_generation()
    test_report_repairs_malformed_llm_output()
    test_incomplete_session_rejected()
    test_strength_detection()
    test_risk_detection()
    test_information_gap_detection()
    test_contradiction_detection()
    test_non_genuine_contradiction_filtered()
    test_category_scores_in_range()
    test_overall_score_bounds_and_determinism()
    test_risk_level_thresholds()
    test_idempotency()
    test_get_report_not_found()
    test_get_report_after_complete()
    test_api_endpoints()

    print("\nAll Phase 3 tests passed.")


if __name__ == "__main__":
    main()
