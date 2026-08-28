"""Decide whether an answer needs a follow-up question, and generate it if so.

Reuses the same "prompt for raw JSON, parse/validate manually" approach as
app/question_engine/generator.py, since the NVIDIA-hosted model doesn't
reliably support LangChain's tool-calling-based structured output.
"""

import json
import re
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config import LLM_MODEL_NAME, LLM_TEMPERATURE, require_nvidia_api_key
from app.graph.repair_graph import run_repair_graph
from app.question_engine.context_builder import FollowupContext, render_followup_prompt_context

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class FollowupDecision(BaseModel):
    follow_up_required: bool
    question: str | None = Field(default=None)
    category: str | None = Field(default=None)
    priority: str | None = Field(default=None)
    reason: str = Field(description="Why a follow-up is or isn't needed.")


class SupportsInvoke(Protocol):
    """Minimal interface the followup generator needs from an LLM chain —
    satisfied by both ChatNVIDIA and test stubs.
    """

    def invoke(self, inputs: dict) -> object: ...


SYSTEM_PROMPT = """\
You are a due-diligence analyst conducting a live, interactive investor \
questioning session with a startup founder.

You will be given the startup's information, its stage, relevant \
due-diligence reference material, the original top-10 question this turn \
belongs to, the current question actually asked, the founder's current \
answer, and the prior questions/answers/follow-ups already covered in this \
session.

Your job is to decide whether ONE meaningful follow-up question is needed \
before moving on.

Ask a follow-up ONLY if at least one of these is true:
- The answer is vague, incomplete, or doesn't actually address the question.
- The answer reveals a risk, red flag, or concentration/dependency issue \
worth probing deeper.
- The answer is ambiguous and needs clarification.
- The answer contradicts the startup information given, the reference \
material, or something said earlier in this session — in that case, ask a \
clarification question that names the specific discrepancy.
- There is a genuinely important, non-obvious next question a sharp \
investor would ask based specifically on what was just said.

Do NOT ask a follow-up if:
- The answer already sufficiently addresses the current question.
- The follow-up would repeat the current question, a previous question, or \
an existing follow-up (see the lists provided) in substance, even if worded \
differently.
- The follow-up would ask for information already clearly given anywhere \
in the startup information or the session so far.
- The only value of the follow-up is generic curiosity, not something a \
real investor would need to make a decision.

If unsure, prefer NOT asking a follow-up. Only ask when it would add real value.

When a follow-up is warranted, it must be grounded in the founder's actual \
answer (and, where relevant, the reference material or startup info) — \
never a generic or random question.

Output format:
Respond with ONLY a single valid JSON object, no markdown code fences, no \
prose before or after. The JSON must have this exact shape:

{{
  "follow_up_required": true | false,
  "question": "..." or null,
  "category": "..." or null,
  "priority": "High" | "Medium" | "Low" or null,
  "reason": "..."
}}

If follow_up_required is false, question, category, and priority must all be null.
"""

USER_PROMPT = """\
{context}

Decide whether a follow-up question is needed, following the rules above. \
Respond with ONLY the JSON object.
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
        _llm = ChatNVIDIA(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    return _llm


def _parse_followup_decision(raw_text: str) -> FollowupDecision:
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
        return FollowupDecision.model_validate(data)
    except ValidationError as exc:
        raise RuntimeError(
            f"LLM response JSON did not match the expected schema: {exc}. "
            f"Raw response:\n{raw_text}"
        ) from exc


def _is_duplicate(candidate: str, existing: list[str]) -> bool:
    normalized_candidate = " ".join(candidate.lower().split())
    for q in existing:
        if normalized_candidate == " ".join(q.lower().split()):
            return True
    return False


def decide_followup(ctx: FollowupContext, llm: SupportsInvoke | None = None) -> FollowupDecision:
    """Ask the LLM whether a follow-up is needed for this answer, and return
    its decision. Guards against duplicates as a defensive backstop even
    though the prompt already instructs the LLM to avoid them.

    `llm` is called directly with the rendered prompt messages (rather than
    composed via `_prompt | llm`) so a plain test stub exposing only
    `.invoke()` can stand in for the real ChatNVIDIA runnable.
    """
    target = llm if llm is not None else get_llm()
    decision = run_repair_graph(
        target,
        _prompt,
        {"context": render_followup_prompt_context(ctx)},
        _parse_followup_decision,
    )

    if decision.follow_up_required and decision.question:
        already_asked = [ctx.current_question, *ctx.existing_followup_questions] + [
            t.question for t in ctx.previous_turns
        ]
        if _is_duplicate(decision.question, already_asked):
            return FollowupDecision(
                follow_up_required=False,
                question=None,
                category=None,
                priority=None,
                reason=(
                    "Suppressed: proposed follow-up duplicated a question "
                    "already asked in this session."
                ),
            )

    return decision
