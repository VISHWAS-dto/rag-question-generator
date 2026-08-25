"""Orchestrates the full Phase 3 pipeline: collect the completed interview,
retrieve RAG evidence, run LLM analysis, apply deterministic risk backstops,
compute deterministic scores, and assemble the final AssessmentReportSchema.

This module contains no LLM prompting itself — it wires together
evidence.py, analyzer.py, risk_analyzer.py, and scorer.py, and is the single
entry point app/session_manager.py calls to generate a report.
"""

import uuid
from datetime import datetime, timezone

from app.models import AssessmentSession
from app.report_engine.analyzer import SupportsInvoke, analyze_interview
from app.report_engine.evidence import retrieve_interview_evidence
from app.report_engine.risk_analyzer import apply_risk_backstops
from app.report_engine.schemas import AssessmentReportSchema
from app.report_engine.scorer import compute_overall_score, determine_risk_level, score_categories


class ReportGenerationError(RuntimeError):
    """Raised when any stage of report generation fails (RAG retrieval, LLM
    call, or malformed/invalid LLM output) so the caller can return a clear
    HTTP error instead of storing a malformed report.
    """


def generate_report(session: AssessmentSession, llm: SupportsInvoke | None = None) -> AssessmentReportSchema:
    """Run the full Phase 3 pipeline for a completed session and return the
    validated report. Does not touch the database — the caller
    (session_manager.complete_session) is responsible for persisting it.
    """
    try:
        rag_context = retrieve_interview_evidence(session)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear, typed error
        raise ReportGenerationError(f"RAG evidence retrieval failed: {exc}") from exc

    try:
        analysis = analyze_interview(session, rag_context, llm=llm)
    except Exception as exc:  # noqa: BLE001
        raise ReportGenerationError(f"Interview analysis failed: {exc}") from exc

    analysis = apply_risk_backstops(analysis)

    category_scores = score_categories(analysis.category_assessments)
    overall_score = compute_overall_score(category_scores)
    risk_level = determine_risk_level(overall_score, analysis.risks)

    try:
        return AssessmentReportSchema(
            report_id=uuid.uuid4().hex,
            session_id=session.session_id,
            overall_score=overall_score,
            risk_level=risk_level,
            executive_summary=analysis.executive_summary,
            strengths=analysis.strengths,
            risks=analysis.risks,
            information_gaps=analysis.information_gaps,
            contradictions=analysis.contradictions,
            category_scores={cat.value: score for cat, score in category_scores.items()},
            recommendations=analysis.recommendations,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:  # noqa: BLE001
        raise ReportGenerationError(f"Report assembly/validation failed: {exc}") from exc
