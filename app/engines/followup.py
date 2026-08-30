"""Decide whether a founder's answer needs one follow-up question.

Refactored from app/question_engine/followup.py + context_builder.py. The
context assembly (what the LLM sees) is a pure function of already-loaded
data; retrieval goes through `RAGClient`, the model through `LLMClient`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.logging import get_logger

from app.clients.llm import LLMClient, LLMError, Message
from app.clients.rag import RAGClient, RAGUnavailableError, format_context
from app.domain.parsing import RawOutputError, parse_json_object
from app.domain.schemas import FollowupDecision
from app.engines.prompts import FOLLOWUP_SYSTEM_PROMPT, FOLLOWUP_USER_PROMPT
from app.llm_repair import LLMOutputError, run_with_repair

log = get_logger("app.engines.followup")


class FollowupDecisionError(RuntimeError):
    """Any failure while deciding on a follow-up (RAG, LLM, or parsing)."""


@dataclass
class PriorTurn:
    question: str
    answer: str


@dataclass
class FollowupContext:
    startup_info: str
    startup_stage: str | None
    rag_context: str
    top_level_question: str
    current_question: str
    current_answer: str
    previous_turns: list[PriorTurn] = field(default_factory=list)
    existing_followup_questions: list[str] = field(default_factory=list)


def render_followup_context(ctx: FollowupContext) -> str:
    lines: list[str] = [
        "STARTUP INFORMATION:",
        ctx.startup_info,
        "",
        f"STARTUP STAGE: {ctx.startup_stage or 'Not specified'}",
        "",
        "RELEVANT DUE-DILIGENCE REFERENCE MATERIAL:",
        ctx.rag_context or "(none retrieved)",
        "",
    ]
    if ctx.previous_turns:
        lines.append("PREVIOUS QUESTIONS AND ANSWERS IN THIS SESSION:")
        for i, turn in enumerate(ctx.previous_turns, start=1):
            lines.append(f"{i}. Q: {turn.question}")
            lines.append(f"   A: {turn.answer}")
        lines.append("")
    if ctx.existing_followup_questions:
        lines.append("FOLLOW-UP QUESTIONS ALREADY ASKED FOR THE CURRENT TOP-10 QUESTION:")
        for i, q in enumerate(ctx.existing_followup_questions, start=1):
            lines.append(f"{i}. {q}")
        lines.append("")
    lines += [
        "ORIGINAL TOP-10 QUESTION THIS TURN BELONGS TO:",
        ctx.top_level_question,
        "",
        "CURRENT QUESTION:",
        ctx.current_question,
        "",
        "CURRENT ANSWER:",
        ctx.current_answer,
    ]
    return "\n".join(lines)


def retrieve_followup_context(
    rag: RAGClient, *, current_question: str, current_answer: str, top_k: int, collection: str
) -> str:
    query = f"{current_question}\n{current_answer}"
    try:
        chunks = rag.retrieve(query, top_k=top_k, collection=collection)
    except RAGUnavailableError as exc:
        raise FollowupDecisionError(f"Retrieval unavailable: {exc}") from exc
    return format_context(chunks)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _is_duplicate(candidate: str, existing: list[str]) -> bool:
    norm = _normalize(candidate)
    return any(norm == _normalize(q) for q in existing)


def decide_followup(
    *,
    llm: LLMClient,
    ctx: FollowupContext,
    temperature: float,
    max_tokens: int,
    max_repair_attempts: int,
    timeout: float | None = None,
) -> FollowupDecision:
    messages = [
        Message(role="system", content=FOLLOWUP_SYSTEM_PROMPT),
        Message(
            role="user",
            content=FOLLOWUP_USER_PROMPT.format(context=render_followup_context(ctx)),
        ),
    ]

    try:
        decision = run_with_repair(
            llm,
            messages,
            lambda raw: parse_json_object(raw, FollowupDecision),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_repair_attempts=max_repair_attempts,
        )
    except (LLMOutputError, RawOutputError) as exc:
        raise FollowupDecisionError(f"Follow-up decision produced unusable output: {exc}") from exc
    except LLMError as exc:
        raise FollowupDecisionError(f"LLM endpoint failed during follow-up decision: {exc}") from exc

    if decision.follow_up_required and decision.question:
        already_asked = [
            ctx.current_question,
            *ctx.existing_followup_questions,
            *(t.question for t in ctx.previous_turns),
        ]
        if _is_duplicate(decision.question, already_asked):
            return FollowupDecision(
                follow_up_required=False,
                question=None,
                category=None,
                priority=None,
                reason="Suppressed: proposed follow-up duplicated a question already asked "
                "in this session.",
            )

    return decision
