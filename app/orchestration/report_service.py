"""Assemble the final Phase 3 report: analyze the interview (LLM), apply
deterministic risk backstops, compute deterministic scores, validate.

Ported from app/report_engine/report_generator.py, adapted to the injected
engine signature. Contains no prompting itself.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from shared.config import AppSettings
from shared.logging import get_logger

from app.clients.llm import LLMClient
from app.clients.rag import RAGClient
from app.domain.risk_backstops import apply_risk_backstops
from app.domain.schemas import AssessmentReportSchema
from app.domain.scoring import (
    ScoringError,
    compute_overall_score,
    determine_risk_level,
    score_categories,
)
from app.engines.analyzer import (
    InterviewAnalysisError,
    InterviewTurn,
    analyze_interview,
    retrieve_evidence,
)

log = get_logger("app.orchestration.report")


class ReportGenerationError(RuntimeError):
    """Any stage of report generation failed (RAG, LLM, scoring, or validation)."""


def build_report(
    *,
    llm: LLMClient,
    rag: RAGClient,
    settings: AppSettings,
    session_id: str,
    startup_info: str,
    startup_stage: str | None,
    turns: list[InterviewTurn],
) -> AssessmentReportSchema:
    try:
        rag_context = retrieve_evidence(
            rag,
            startup_info=startup_info,
            turns=turns,
            top_k=settings.rag_retrieval_top_k,
            collection=settings.rag_collection,
        )
    except InterviewAnalysisError as exc:
        raise ReportGenerationError(f"RAG evidence retrieval failed: {exc}") from exc

    try:
        analysis = analyze_interview(
            llm=llm,
            startup_info=startup_info,
            startup_stage=startup_stage,
            rag_context=rag_context,
            turns=turns,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_report_max_tokens,
            max_repair_attempts=settings.llm_max_repair_attempts,
        )
    except InterviewAnalysisError as exc:
        raise ReportGenerationError(f"Interview analysis failed: {exc}") from exc

    analysis = apply_risk_backstops(analysis)

    try:
        category_scores = score_categories(analysis.category_assessments)
        overall_score = compute_overall_score(category_scores)
        risk_level = determine_risk_level(overall_score, analysis.risks)
    except ScoringError as exc:
        raise ReportGenerationError(f"Deterministic scoring failed: {exc}") from exc

    try:
        return AssessmentReportSchema(
            report_id=uuid.uuid4().hex,
            session_id=session_id,
            overall_score=overall_score,
            risk_level=risk_level,
            executive_summary=analysis.executive_summary,
            strengths=analysis.strengths,
            risks=analysis.risks,
            information_gaps=analysis.information_gaps,
            contradictions=analysis.contradictions,
            category_scores={cat.value: score for cat, score in category_scores.items()},
            recommendations=analysis.recommendations,
            created_at=datetime.now(UTC).isoformat(),
        )
    except Exception as exc:
        raise ReportGenerationError(f"Report assembly/validation failed: {exc}") from exc
