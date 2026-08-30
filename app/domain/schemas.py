"""Pydantic schemas: the question-generation output, the follow-up decision,
and the full Phase 3 report contract.

`InterviewAnalysis` is the contract between the LLM analysis step and the rest
of the system: the model's raw JSON is validated against it, and the final
`AssessmentReportSchema` is assembled in Python by combining that analysis
with deterministically-computed scores (see `app/domain/scoring.py`). The LLM
never sets the overall score or risk level.

Consolidated from the original app/question_engine and app/report_engine
schema modules, unchanged in shape.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Shared enums
# --------------------------------------------------------------------------- #


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Priority(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Category(StrEnum):
    MARKET = "market"
    PRODUCT = "product"
    TRACTION = "traction"
    BUSINESS_MODEL = "business_model"
    FINANCIALS = "financials"
    TEAM = "team"
    COMPETITION = "competition"
    TECHNOLOGY = "technology"
    GO_TO_MARKET = "go_to_market"
    RISK = "risk"


class EvidenceSource(StrEnum):
    FOUNDER_ANSWER = "FOUNDER_ANSWER"
    KNOWLEDGE_BASE = "KNOWLEDGE_BASE"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"


# --------------------------------------------------------------------------- #
# Phase 1: question generation
# --------------------------------------------------------------------------- #


class DueDiligenceQuestion(BaseModel):
    question: str = Field(min_length=1)
    category: str = Field(min_length=1)
    priority: str = Field(description="One of: High, Medium, Low")
    reason: str = Field(min_length=1)
    source_context: str = Field(min_length=1)


class TopQuestions(BaseModel):
    questions: list[DueDiligenceQuestion]


# --------------------------------------------------------------------------- #
# Phase 2: follow-up decision
# --------------------------------------------------------------------------- #


class FollowupDecision(BaseModel):
    follow_up_required: bool
    question: str | None = None
    category: str | None = None
    priority: str | None = None
    reason: str = Field(description="Why a follow-up is or isn't needed.")


# --------------------------------------------------------------------------- #
# Phase 3: report
# --------------------------------------------------------------------------- #


class Evidence(BaseModel):
    source: EvidenceSource
    detail: str


class Strength(BaseModel):
    title: str
    description: str
    category: Category
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: str = Field(description="One of: Low, Medium, High")


class Risk(BaseModel):
    title: str
    description: str
    category: Category
    severity: Severity
    impact: str
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: str = Field(description="One of: Low, Medium, High")


class InformationGap(BaseModel):
    topic: str
    why_it_matters: str
    priority: Priority


class Contradiction(BaseModel):
    topic: str
    earlier_claim: str
    later_claim: str
    explanation: str
    severity: Severity


class Recommendation(BaseModel):
    action: str
    reason: str
    priority: Priority


class CategoryAssessment(BaseModel):
    category: Category
    assessment: str = Field(description="Excellent | Strong | Moderate | Weak | Critical")
    rationale: str
    evidence_strength: str = Field(description="One of: Low, Medium, High")
    evidence_gaps: list[str] = Field(default_factory=list)


class InterviewAnalysis(BaseModel):
    executive_summary: str
    strengths: list[Strength] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    information_gaps: list[InformationGap] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    category_assessments: list[CategoryAssessment]
    recommendations: list[Recommendation] = Field(default_factory=list)

    @field_validator("category_assessments")
    @classmethod
    def _all_categories_present(cls, v: list[CategoryAssessment]) -> list[CategoryAssessment]:
        missing = set(Category) - {c.category for c in v}
        if missing:
            raise ValueError(f"Missing category assessments for: {sorted(m.value for m in missing)}")
        return v


class AssessmentReportSchema(BaseModel):
    report_id: str
    session_id: str
    overall_score: float = Field(ge=0, le=10)
    risk_level: RiskLevel
    executive_summary: str
    strengths: list[Strength]
    risks: list[Risk]
    information_gaps: list[InformationGap]
    contradictions: list[Contradiction]
    category_scores: dict[Category, float]
    recommendations: list[Recommendation]
    created_at: str

    @field_validator("category_scores")
    @classmethod
    def _scores_in_range(cls, v: dict[Category, float]) -> dict[Category, float]:
        for category, score in v.items():
            if not (0 <= score <= 10):
                raise ValueError(f"Category score for {category} out of range 0-10: {score}")
        return v
