"""Pydantic schemas for the Phase 3 due-diligence report.

These types are the contract between the LLM analysis step and the rest of
the system: the LLM's raw JSON response is parsed and validated against
`InterviewAnalysis` (everything except the deterministic scores), and the
final `AssessmentReportSchema` is assembled in Python by combining that
analysis with the deterministically-computed category/overall scores and
risk level (see app/report_engine/scorer.py). The LLM never sets the
overall score or risk level directly.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


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
    """Distinguishes what actually backs a finding — see PHASE 5/RAG EVIDENCE
    in the spec: founder-provided evidence, knowledge-base guidance, model
    inference, or an acknowledged absence of evidence.
    """

    FOUNDER_ANSWER = "FOUNDER_ANSWER"
    KNOWLEDGE_BASE = "KNOWLEDGE_BASE"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"


class Evidence(BaseModel):
    source: EvidenceSource
    detail: str = Field(description="The specific quote, fact, or gap this evidence points to.")


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
    impact: str = Field(description="What this risk means for the investment decision.")
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
    """The LLM's evidence-based judgment for one category, BEFORE the
    deterministic 0-10 score is computed in Python. `evidence_strength`
    drives that computation (see scorer.py) rather than letting the LLM
    hand back a numeric score directly.
    """

    category: Category
    assessment: str = Field(description="Excellent | Strong | Moderate | Weak | Critical")
    rationale: str
    evidence_strength: str = Field(description="One of: Low, Medium, High")
    evidence_gaps: list[str] = Field(default_factory=list)


class InterviewAnalysis(BaseModel):
    """The full structured output the LLM must produce from one interview.
    Validated immediately after parsing; the overall score and risk level
    are deliberately NOT part of this schema — those are computed
    deterministically afterward in scorer.py.
    """

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
        seen = {c.category for c in v}
        missing = set(Category) - seen
        if missing:
            raise ValueError(f"Missing category assessments for: {sorted(missing)}")
        return v


class AssessmentReportSchema(BaseModel):
    """The final, fully-assembled Phase 3 report — what gets stored and
    returned by the API.
    """

    report_id: str
    session_id: str
    overall_score: float = Field(ge=0, le=10)
    risk_level: RiskLevel
    executive_summary: str
    strengths: list[Strength]
    risks: list[Risk]
    information_gaps: list[InformationGap]
    contradictions: list[Contradiction]
    category_scores: dict[Category, float] = Field(
        description="0-10 score per category. 9-10 Excellent, 7-8 Strong, "
        "5-6 Moderate, 3-4 Weak, 0-2 Critical."
    )
    recommendations: list[Recommendation]
    created_at: str

    @field_validator("category_scores")
    @classmethod
    def _scores_in_range(cls, v: dict[Category, float]) -> dict[Category, float]:
        for category, score in v.items():
            if not (0 <= score <= 10):
                raise ValueError(f"Category score for {category} out of range 0-10: {score}")
        return v
