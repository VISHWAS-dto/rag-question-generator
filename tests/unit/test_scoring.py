"""Deterministic scoring and risk-level logic (app/domain/scoring.py).

Ports the scoring assertions from tests/test_phase3.py.
"""

from __future__ import annotations

import pytest
from app.domain.schemas import (
    Category,
    CategoryAssessment,
    Risk,
    RiskLevel,
    Severity,
)
from app.domain.scoring import (
    ScoringError,
    compute_overall_score,
    determine_risk_level,
    score_categories,
    score_category,
)

pytestmark = pytest.mark.unit


def test_category_scores_always_in_range():
    for assessment in ("Excellent", "Strong", "Moderate", "Weak", "Critical"):
        for strength in ("Low", "Medium", "High"):
            ca = CategoryAssessment(
                category=Category.FINANCIALS,
                assessment=assessment,
                rationale="t",
                evidence_strength=strength,
                evidence_gaps=[],
            )
            assert 0 <= score_category(ca) <= 10


def test_unknown_assessment_tier_raises_scoring_error():
    ca = CategoryAssessment(
        category=Category.TEAM,
        assessment="Superb",  # not in the vocabulary
        rationale="t",
        evidence_strength="High",
    )
    with pytest.raises(ScoringError):
        score_category(ca)


def test_overall_score_bounds_and_determinism():
    assert compute_overall_score({c: 10.0 for c in Category}) == 10.0
    assert compute_overall_score({c: 0.0 for c in Category}) == 0.0
    assert compute_overall_score({c: 5.0 for c in Category}) == 5.0
    mixed = {c: 5.0 for c in Category}
    assert compute_overall_score(mixed) == compute_overall_score(mixed)


def test_risk_level_thresholds():
    assert determine_risk_level(9.0, []) == RiskLevel.LOW
    assert determine_risk_level(8.0, []) == RiskLevel.LOW
    assert determine_risk_level(7.9, []) == RiskLevel.MEDIUM
    assert determine_risk_level(6.0, []) == RiskLevel.MEDIUM
    assert determine_risk_level(5.9, []) == RiskLevel.HIGH
    assert determine_risk_level(4.0, []) == RiskLevel.HIGH
    assert determine_risk_level(3.9, []) == RiskLevel.CRITICAL
    assert determine_risk_level(0.0, []) == RiskLevel.CRITICAL


def test_critical_risk_overrides_high_numeric_score():
    critical = Risk(
        title="Unresolved litigation",
        description="Active lawsuit threatens core IP.",
        category=Category.RISK,
        severity=Severity.CRITICAL,
        impact="Could invalidate the core product.",
        evidence=[],
        confidence="High",
    )
    assert determine_risk_level(9.5, [critical]) == RiskLevel.CRITICAL


def test_score_categories_maps_all():
    assessments = [
        CategoryAssessment(
            category=c,
            assessment="Moderate",
            rationale="t",
            evidence_strength="Medium",
        )
        for c in Category
    ]
    scores = score_categories(assessments)
    assert set(scores) == set(Category)
