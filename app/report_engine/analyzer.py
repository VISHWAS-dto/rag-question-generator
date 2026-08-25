"""Analyzes a complete interview and produces structured, evidence-based
findings (strengths, risks, information gaps, category assessments,
recommendations) — everything in AssessmentReportSchema except the
deterministic overall score and risk level.

Uses the same "prompt for raw JSON, parse/validate manually" approach as
app/question_engine/generator.py and app/question_engine/followup.py, since
the NVIDIA-hosted model doesn't reliably support LangChain's tool-calling-
based structured output.
"""

import json
import re
from typing import Protocol

from pydantic import ValidationError

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    REPORT_LLM_MAX_TOKENS,
    require_nvidia_api_key,
)
from app.models import AssessmentSession
from app.report_engine.evidence import InterviewTurn, collect_interview_turns
from app.report_engine.schemas import Category, InterviewAnalysis

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

CATEGORY_LIST = ", ".join(c.value for c in Category)


class SupportsInvoke(Protocol):
    """Minimal interface the analyzer needs from an LLM chain — satisfied by
    both ChatNVIDIA and test stubs.
    """

    def invoke(self, inputs: dict) -> object: ...


SYSTEM_PROMPT = f"""\
You are a senior due-diligence analyst producing a structured investment \
assessment from a completed founder interview.

You will be given the startup's self-reported information, its stage, the \
full interview transcript (Top-10 due-diligence questions, any follow-ups, \
and the founder's answers to each), and relevant due-diligence reference \
material retrieved from a knowledge base.

CRITICAL RULES — evidence discipline:
- Use ONLY information that is actually present in the interview transcript \
or the retrieved reference material. Never invent facts, metrics, or \
benchmarks that are not present in what you were given.
- Distinguish founder CLAIMS from VERIFIED evidence. A founder's statement \
is a claim, not proof, unless it comes with concrete, checkable detail \
(numbers, named metrics, specifics).
- Do NOT treat the absence of information as positive. If a founder claims \
something ("we have excellent retention") without providing supporting \
data, that is a claim with an evidence gap — not a strength, and it should \
lower your confidence, not raise it.
- Every strength and risk must cite evidence: quote or closely paraphrase \
the specific founder answer, or the specific knowledge-base guidance, that \
supports it. For each Evidence item, set "source" to exactly one of:
  - "FOUNDER_ANSWER" — a specific answer the founder gave
  - "KNOWLEDGE_BASE" — the retrieved due-diligence reference material
  - "MODEL_INFERENCE" — a reasonable inference you are drawing, clearly \
labeled as such (use sparingly, only when directly supported by the above)
  - "MISSING_EVIDENCE" — used to document that expected information is \
absent (e.g. for information_gaps or a risk grounded in an evidence gap)
- Only report a genuine contradiction when two answers actually conflict \
(e.g. different numbers for the same metric). Do not invent contradictions.
- Only report an information gap when the interview indicates that specific \
information is missing or insufficiently detailed — do not dump a generic \
checklist of every due-diligence topic regardless of relevance.
- Assign every one of these categories a CategoryAssessment, even if the \
interview said little about it (in that case, assessment should reflect the \
lack of coverage, e.g. "Weak" or "Critical" with evidence_strength "Low", \
and evidence_gaps listing what's missing): {CATEGORY_LIST}
- Do NOT invent a numeric score. For category_assessments, set "assessment" \
to one of Excellent/Strong/Moderate/Weak/Critical based on the evidence \
actually present, and "evidence_strength" to Low/Medium/High reflecting how \
well-supported that judgment is (Low if you are relying mostly on \
unverified founder claims).

Output format:
Respond with ONLY a single valid JSON object, no markdown code fences, no \
prose before or after. The JSON must match this exact shape:

{{{{
  "executive_summary": "2-4 sentence overview of the startup's overall \
investment readiness",
  "strengths": [
    {{{{"title": "...", "description": "...", "category": "...", \
"evidence": [{{{{"source": "...", "detail": "..."}}}}], \
"confidence": "Low" | "Medium" | "High"}}}}
  ],
  "risks": [
    {{{{"title": "...", "description": "...", "category": "...", \
"severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL", "impact": "...", \
"evidence": [{{{{"source": "...", "detail": "..."}}}}], \
"confidence": "Low" | "Medium" | "High"}}}}
  ],
  "information_gaps": [
    {{{{"topic": "...", "why_it_matters": "...", \
"priority": "LOW" | "MEDIUM" | "HIGH"}}}}
  ],
  "contradictions": [
    {{{{"topic": "...", "earlier_claim": "...", "later_claim": "...", \
"explanation": "...", "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"}}}}
  ],
  "category_assessments": [
    {{{{"category": "market" | "product" | "traction" | "business_model" | \
"financials" | "team" | "competition" | "technology" | "go_to_market" | \
"risk", "assessment": "Excellent" | "Strong" | "Moderate" | "Weak" | \
"Critical", "rationale": "...", "evidence_strength": "Low" | "Medium" | \
"High", "evidence_gaps": ["..."]}}}}
  ],
  "recommendations": [
    {{{{"action": "specific, actionable next step — not generic", \
"reason": "...", "priority": "LOW" | "MEDIUM" | "HIGH"}}}}
  ]
}}}}
"""

USER_PROMPT = """\
STARTUP INFORMATION:
{startup_info}

STARTUP STAGE: {startup_stage}

RELEVANT DUE-DILIGENCE REFERENCE MATERIAL:
{rag_context}

FULL INTERVIEW TRANSCRIPT:
{transcript}

Analyze this interview following the rules above. Respond with ONLY the JSON object.
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]
)

_llm = None


def get_llm() -> ChatNVIDIA:
    global _llm
    if _llm is None:
        require_nvidia_api_key()
        _llm = ChatNVIDIA(
            model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, max_tokens=REPORT_LLM_MAX_TOKENS
        )
    return _llm


def render_transcript(turns: list[InterviewTurn]) -> str:
    lines: list[str] = []
    for i, turn in enumerate(turns, start=1):
        kind = "Follow-up" if turn.is_followup else "Top-10"
        lines.append(f"{i}. [{kind}/{turn.category or 'Uncategorized'}] Q: {turn.question}")
        lines.append(f"   A: {turn.answer}")
    return "\n".join(lines) if lines else "(no answered questions)"


def _parse_analysis(raw_text: str) -> InterviewAnalysis:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise RuntimeError(
            f"LLM response did not contain a JSON object. Raw response:\n{raw_text}"
        )

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM response was not valid JSON: {exc}. Raw response:\n{raw_text}"
        ) from exc

    try:
        return InterviewAnalysis.model_validate(data)
    except ValidationError as exc:
        raise RuntimeError(
            f"LLM response JSON did not match the expected schema: {exc}. "
            f"Raw response:\n{raw_text}"
        ) from exc


def analyze_interview(
    session: AssessmentSession, rag_context: str, llm: SupportsInvoke | None = None
) -> InterviewAnalysis:
    """Run the full-interview analysis and return validated, structured findings.

    `llm` is called directly with the rendered prompt messages (rather than
    composed via `_prompt | llm`), mirroring app/question_engine/followup.py,
    so a plain test stub exposing only `.invoke()` can stand in for the real
    ChatNVIDIA runnable.
    """
    turns = collect_interview_turns(session)
    transcript = render_transcript(turns)

    target = llm if llm is not None else get_llm()
    messages = _prompt.invoke(
        {
            "startup_info": session.startup_info,
            "startup_stage": session.startup_stage or "Not specified",
            "rag_context": rag_context or "(none retrieved)",
            "transcript": transcript,
        }
    )
    response = target.invoke(messages)
    return _parse_analysis(response.content)
