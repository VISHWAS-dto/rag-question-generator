"""Deterministic backstops over the LLM's risk/contradiction findings.

Phase 2's contradiction detection is folded into decide_followup's single
LLM call (app/question_engine/followup.py) rather than being a standalone
function — there is no separate deterministic contradiction detector to
import. Phase 3 reuses the same *approach*: ask the LLM (here, as part of
analyze_interview) to find contradictions grounded in the actual transcript,
then apply a defensive, deterministic backstop here, the same way
followup.py's decide_followup backstops the LLM with _is_duplicate.
"""

from app.report_engine.schemas import Contradiction, InterviewAnalysis, Risk, Severity


def _is_genuine_contradiction(c: Contradiction) -> bool:
    """Defensive backstop: drop a claimed contradiction if the two claims
    given are actually identical (a common LLM failure mode: flagging a
    restatement as a contradiction). Whitespace/case-insensitive compare.
    """
    earlier = " ".join(c.earlier_claim.lower().split())
    later = " ".join(c.later_claim.lower().split())
    return earlier != later


def filter_contradictions(contradictions: list[Contradiction]) -> list[Contradiction]:
    return [c for c in contradictions if _is_genuine_contradiction(c)]


def escalate_risks_from_contradictions(
    risks: list[Risk], contradictions: list[Contradiction]
) -> list[Risk]:
    """A CRITICAL-severity contradiction (e.g. materially different revenue
    figures) is itself a due-diligence risk, even if the LLM didn't also
    surface it as one. This does not fabricate detail: it only escalates
    what the LLM already found in `contradictions`, using the contradiction's
    own fields as the risk's evidence.
    """
    from app.report_engine.schemas import Category, Evidence, EvidenceSource

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
    """Apply deterministic backstops to the raw LLM analysis: filter
    non-genuine contradictions, then escalate any severe contradiction into
    an explicit risk so it's visible in the risks list, not just buried in
    contradictions.
    """
    contradictions = filter_contradictions(analysis.contradictions)
    risks = escalate_risks_from_contradictions(analysis.risks, contradictions)
    return analysis.model_copy(update={"contradictions": contradictions, "risks": risks})
