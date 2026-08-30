"""Analyze a completed interview into structured, evidence-based findings.

Refactored from app/report_engine/analyzer.py. Produces everything in
`InterviewAnalysis` (not the deterministic scores). Transcript rendering is a
pure function of already-loaded turns; retrieval via `RAGClient`, model via
`LLMClient`.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.logging import get_logger

from app.clients.llm import LLMClient, LLMError, Message
from app.clients.rag import RAGClient, RAGUnavailableError, format_context
from app.domain.parsing import RawOutputError, parse_json_object
from app.domain.schemas import InterviewAnalysis
from app.engines.prompts import ANALYSIS_USER_PROMPT, analysis_system_prompt
from app.llm_repair import LLMOutputError, run_with_repair

log = get_logger("app.engines.analyzer")


class InterviewAnalysisError(RuntimeError):
    """Any failure while analyzing the interview (RAG, LLM, or parsing)."""


@dataclass
class InterviewTurn:
    question: str
    category: str | None
    is_followup: bool
    answer: str


def render_transcript(turns: list[InterviewTurn]) -> str:
    if not turns:
        return "(no answered questions)"
    lines: list[str] = []
    for i, turn in enumerate(turns, start=1):
        kind = "Follow-up" if turn.is_followup else "Top-10"
        lines.append(f"{i}. [{kind}/{turn.category or 'Uncategorized'}] Q: {turn.question}")
        lines.append(f"   A: {turn.answer}")
    return "\n".join(lines)


def retrieve_evidence(
    rag: RAGClient,
    *,
    startup_info: str,
    turns: list[InterviewTurn],
    top_k: int,
    collection: str,
) -> str:
    query = "\n".join([startup_info, *(t.answer for t in turns)])
    try:
        chunks = rag.retrieve(query, top_k=top_k, collection=collection)
    except RAGUnavailableError as exc:
        raise InterviewAnalysisError(f"Evidence retrieval unavailable: {exc}") from exc
    return format_context(chunks)


def analyze_interview(
    *,
    llm: LLMClient,
    startup_info: str,
    startup_stage: str | None,
    rag_context: str,
    turns: list[InterviewTurn],
    temperature: float,
    max_tokens: int,
    max_repair_attempts: int,
    timeout: float | None = None,
) -> InterviewAnalysis:
    messages = [
        Message(role="system", content=analysis_system_prompt()),
        Message(
            role="user",
            content=ANALYSIS_USER_PROMPT.format(
                startup_info=startup_info,
                startup_stage=startup_stage or "Not specified",
                rag_context=rag_context or "(none retrieved)",
                transcript=render_transcript(turns),
            ),
        ),
    ]

    try:
        return run_with_repair(
            llm,
            messages,
            lambda raw: parse_json_object(raw, InterviewAnalysis),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_repair_attempts=max_repair_attempts,
        )
    except (LLMOutputError, RawOutputError) as exc:
        raise InterviewAnalysisError(f"Interview analysis produced unusable output: {exc}") from exc
    except LLMError as exc:
        raise InterviewAnalysisError(f"LLM endpoint failed during interview analysis: {exc}") from exc
