"""Deterministic backstops over the LLM's risk / contradiction findings.

Unchanged from the original app/report_engine/risk_analyzer.py: drop
non-genuine contradictions (identical claims), and escalate any HIGH/CRITICAL
contradiction into an explicit risk so it is visible in the risks list.
"""

from __future__ import annotations

from app.domain.schemas import (
    Category,
    Contradiction,
    Evidence,
    EvidenceSource,
    InterviewAnalysis,
    Risk,
    Severity,
)


def _is_genuine_contradiction(c: Contradiction) -> bool:
    earlier = " ".join(c.earlier_claim.lower().split())
    later = " ".join(c.later_claim.lower().split())
    return earlier != later


def filter_contradictions(contradictions: list[Contradiction]) -> list[Contradiction]:
    return [c for c in contradictions if _is_genuine_contradiction(c)]


def escalate_risks_from_contradictions(
    risks: list[Risk], contradictions: list[Contradiction]
) -> list[Risk]:
    existing_titles = {r.title.lower() for r in risks}
    escalated = list(risks)

    for c in contradictions:
        if c.severity not in (Severity.HIGH, Severity.CRITICAL):
            continue
        title = f"Inconsistent claims: {c.topic}"
        if title.lower() in existing_titles:
            continue
        escalated.append(
            Risk(
                title=title,
                description=c.explanation,
                category=Category.RISK,
                severity=c.severity,
                impact="Inconsistent figures undermine confidence in founder-reported data "
                "and require verification before the claim can be relied upon.",
                evidence=[
                    Evidence(source=EvidenceSource.FOUNDER_ANSWER, detail=c.earlier_claim),
                    Evidence(source=EvidenceSource.FOUNDER_ANSWER, detail=c.later_claim),
                ],
                confidence="High",
            )
        )
        existing_titles.add(title.lower())

    return escalated


def apply_risk_backstops(analysis: InterviewAnalysis) -> InterviewAnalysis:
    contradictions = filter_contradictions(analysis.contradictions)
    risks = escalate_risks_from_contradictions(analysis.risks, contradictions)
    return analysis.model_copy(update={"contradictions": contradictions, "risks": risks})
