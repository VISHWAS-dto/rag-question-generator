"""Deterministic scoring: converts the LLM's qualitative CategoryAssessments
into 0-10 category scores, aggregates them into an overall score, and
determines the risk level.

The LLM never assigns a numeric score directly (see analyzer.py) — scoring
is done here, in Python, from the qualitative assessment + evidence strength
the LLM provided, plus critical risks detected in the interview. This keeps
scores auditable and reproducible instead of an opaque LLM number.
"""

from app.report_engine.schemas import (
    Category,
    CategoryAssessment,
    Risk,
    RiskLevel,
    Severity,
)

# Base score per qualitative assessment tier (midpoint of each band):
# 9-10 Excellent, 7-8 Strong, 5-6 Moderate, 3-4 Weak, 0-2 Critical.
_ASSESSMENT_BASE_SCORE = {
    "Excellent": 9.5,
    "Strong": 7.5,
    "Moderate": 5.5,
    "Weak": 3.5,
    "Critical": 1.0,
}

# Evidence strength nudges the score within/around its band: strong evidence
# is rewarded slightly, weak evidence is penalized — "lack of evidence
# should reduce confidence" (spec section 4) translates directly into a
# lower score here rather than just a lower "confidence" label.
_EVIDENCE_ADJUSTMENT = {
    "High": 0.4,
    "Medium": 0.0,
    "Low": -0.6,
}

# Overall-score weights per category. Traction and Financials are weighted
# highest since they carry the most concrete, verifiable signal for
# investment decisions; Competition/Technology are weighted lowest as they
# are typically judged more qualitatively at early stages. Weights sum to 1.0.
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

# Risk-level thresholds against the numeric overall score.
_RISK_THRESHOLDS: list[tuple[float, RiskLevel]] = [
    (8.0, RiskLevel.LOW),
    (6.0, RiskLevel.MEDIUM),
    (4.0, RiskLevel.HIGH),
    (0.0, RiskLevel.CRITICAL),
]


def score_category(assessment: CategoryAssessment) -> float:
    """Map one qualitative CategoryAssessment to a 0-10 score, deterministically."""
    base = _ASSESSMENT_BASE_SCORE[assessment.assessment]
    adjustment = _EVIDENCE_ADJUSTMENT[assessment.evidence_strength]
    score = base + adjustment
    return round(min(10.0, max(0.0, score)), 1)


def score_categories(
    assessments: list[CategoryAssessment],
) -> dict[Category, float]:
    return {a.category: score_category(a) for a in assessments}


def compute_overall_score(category_scores: dict[Category, float]) -> float:
    """Weighted average of category scores using CATEGORY_WEIGHTS, rounded to
    one decimal place. Every category in CATEGORY_WEIGHTS must be present in
    category_scores (InterviewAnalysis validation already enforces that all
    ten categories get an assessment).
    """
    total = sum(category_scores[cat] * weight for cat, weight in CATEGORY_WEIGHTS.items())
    return round(total, 1)


def determine_risk_level(overall_score: float, risks: list[Risk]) -> RiskLevel:
    """Deterministic threshold-based risk level, with one override: any
    unresolved CRITICAL-severity risk forces the overall risk level to at
    least CRITICAL, regardless of the numeric score — a single critical red
    flag (e.g. active litigation, no legal entity) should not be masked by
    otherwise-strong category scores.
    """
    if any(r.severity == Severity.CRITICAL for r in risks):
        return RiskLevel.CRITICAL

    for threshold, level in _RISK_THRESHOLDS:
        if overall_score >= threshold:
            return level
    return RiskLevel.CRITICAL
