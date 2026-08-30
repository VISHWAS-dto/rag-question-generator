"""Deterministic scoring and risk-level logic.

Unchanged from the original app/report_engine/scorer.py. The LLM never assigns
a numeric score - it produces a qualitative CategoryAssessment, and this
module maps that to auditable, reproducible 0-10 scores in Python.
"""

from __future__ import annotations

from app.domain.schemas import Category, CategoryAssessment, Risk, RiskLevel, Severity

_ASSESSMENT_BASE_SCORE = {
    "Excellent": 9.5,
    "Strong": 7.5,
    "Moderate": 5.5,
    "Weak": 3.5,
    "Critical": 1.0,
}

_EVIDENCE_ADJUSTMENT = {
    "High": 0.4,
    "Medium": 0.0,
    "Low": -0.6,
}

CATEGORY_WEIGHTS: dict[Category, float] = {
    Category.MARKET: 0.10,
    Category.PRODUCT: 0.10,
    Category.TRACTION: 0.15,
    Category.BUSINESS_MODEL: 0.10,
    Category.FINANCIALS: 0.15,
    Category.TEAM: 0.10,
    Category.COMPETITION: 0.05,
    Category.TECHNOLOGY: 0.05,
    Category.GO_TO_MARKET: 0.10,
    Category.RISK: 0.10,
}

assert abs(sum(CATEGORY_WEIGHTS.values()) - 1.0) < 1e-9, "CATEGORY_WEIGHTS must sum to 1.0"

_RISK_THRESHOLDS: list[tuple[float, RiskLevel]] = [
    (8.0, RiskLevel.LOW),
    (6.0, RiskLevel.MEDIUM),
    (4.0, RiskLevel.HIGH),
    (0.0, RiskLevel.CRITICAL),
]


class ScoringError(ValueError):
    """A CategoryAssessment carried a value outside the known vocabulary."""


def score_category(assessment: CategoryAssessment) -> float:
    try:
        base = _ASSESSMENT_BASE_SCORE[assessment.assessment]
    except KeyError as exc:
        raise ScoringError(
            f"Unknown assessment tier {assessment.assessment!r} for category "
            f"{assessment.category.value}; expected one of {sorted(_ASSESSMENT_BASE_SCORE)}"
        ) from exc
    try:
        adjustment = _EVIDENCE_ADJUSTMENT[assessment.evidence_strength]
    except KeyError as exc:
        raise ScoringError(
            f"Unknown evidence_strength {assessment.evidence_strength!r} for category "
            f"{assessment.category.value}; expected one of {sorted(_EVIDENCE_ADJUSTMENT)}"
        ) from exc
    return round(min(10.0, max(0.0, base + adjustment)), 1)


def score_categories(assessments: list[CategoryAssessment]) -> dict[Category, float]:
    return {a.category: score_category(a) for a in assessments}


def compute_overall_score(category_scores: dict[Category, float]) -> float:
    total = sum(category_scores[cat] * weight for cat, weight in CATEGORY_WEIGHTS.items())
    return round(total, 1)


def determine_risk_level(overall_score: float, risks: list[Risk]) -> RiskLevel:
    if any(r.severity == Severity.CRITICAL for r in risks):
        return RiskLevel.CRITICAL
    for threshold, level in _RISK_THRESHOLDS:
        if overall_score >= threshold:
            return level
    return RiskLevel.CRITICAL
