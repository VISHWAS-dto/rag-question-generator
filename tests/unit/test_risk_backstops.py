"""Deterministic contradiction/risk backstops (app/domain/risk_backstops.py).

Ports the contradiction-filtering and escalation assertions from
tests/test_phase3.py.
"""

from __future__ import annotations

import pytest
from app.domain.risk_backstops import apply_risk_backstops
from app.domain.schemas import (
    Category,
    CategoryAssessment,
    Contradiction,
    InterviewAnalysis,
    Severity,
)

pytestmark = pytest.mark.unit


def _analysis(contradictions, risks=None):
    return InterviewAnalysis(
        executive_summary="x",
        strengths=[],
        risks=risks or [],
        information_gaps=[],
        contradictions=contradictions,
        category_assessments=[
            CategoryAssessment(
                category=c, assessment="Moderate", rationale="t", evidence_strength="Low"
            )
            for c in Category
        ],
        recommendations=[],
    )


def test_identical_claims_are_filtered_out():
    analysis = _analysis(
        [
            Contradiction(
                topic="Annual revenue",
                earlier_claim="5 crore INR annual revenue.",
                later_claim="5 crore INR annual revenue.",
                explanation="spurious",
                severity=Severity.HIGH,
            )
        ]
    )
    result = apply_risk_backstops(analysis)
    assert result.contradictions == []


def test_severe_contradiction_is_escalated_to_a_risk():
    analysis = _analysis(
        [
            Contradiction(
                topic="Annual revenue",
                earlier_claim="Our annual revenue is 5 crore INR.",
                later_claim="Actually, our annual revenue is 3 crore INR.",
                explanation="Revenue figures inconsistent.",
                severity=Severity.HIGH,
            )
        ]
    )
    result = apply_risk_backstops(analysis)
    assert len(result.contradictions) == 1
    assert any("revenue" in r.title.lower() for r in result.risks)


def test_low_severity_contradiction_is_not_escalated():
    analysis = _analysis(
        [
            Contradiction(
                topic="Team size",
                earlier_claim="15 staff",
                later_claim="about 16 staff",
                explanation="minor",
                severity=Severity.LOW,
            )
        ]
    )
    result = apply_risk_backstops(analysis)
    assert result.risks == []
