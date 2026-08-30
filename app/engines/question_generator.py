"""Generate the top-N ranked due-diligence questions from startup info + RAG
context.

Refactored from app/question_engine/generator.py: retrieval goes through
`RAGClient`, the model through `LLMClient` + the repair loop. Same public
result (`list[DueDiligenceQuestion]`, deduped, capped at N) and same failure
semantics (a typed error, not a bare 500).
"""

from __future__ import annotations

from shared.logging import get_logger

from app.clients.llm import LLMClient, LLMError, Message
from app.clients.rag import RAGClient, RAGUnavailableError, format_context
from app.domain.parsing import RawOutputError, parse_json_object
from app.domain.schemas import DueDiligenceQuestion, TopQuestions
from app.engines.prompts import QUESTION_USER_PROMPT, question_system_prompt
from app.llm_repair import LLMOutputError, run_with_repair

log = get_logger("app.engines.questions")


class QuestionGenerationError(RuntimeError):
    """Any failure while generating the top-N questions (RAG, LLM, or parsing)."""


def _dedupe(questions: list[DueDiligenceQuestion]) -> list[DueDiligenceQuestion]:
    seen: set[str] = set()
    unique: list[DueDiligenceQuestion] = []
    for q in questions:
        normalized = " ".join(q.question.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(q)
    return unique


def generate_top_questions(
    *,
    llm: LLMClient,
    rag: RAGClient,
    startup_info: str,
    startup_stage: str | None,
    num_questions: int,
    top_k: int,
    collection: str,
    temperature: float,
    max_tokens: int,
    max_repair_attempts: int,
    timeout: float | None = None,
) -> list[DueDiligenceQuestion]:
    try:
        chunks = rag.retrieve(startup_info, top_k=top_k, collection=collection)
    except RAGUnavailableError as exc:
        raise QuestionGenerationError(f"Retrieval unavailable: {exc}") from exc

    if not chunks:
        raise QuestionGenerationError(
            "No context retrieved from the knowledge base. Has ingestion run?"
        )

    messages = [
        Message(role="system", content=question_system_prompt(num_questions)),
        Message(
            role="user",
            content=QUESTION_USER_PROMPT.format(
                startup_info=startup_info,
                startup_stage=startup_stage or "Not specified",
                context=format_context(chunks),
                num_questions=num_questions,
            ),
        ),
    ]

    try:
        result = run_with_repair(
            llm,
            messages,
            lambda raw: parse_json_object(raw, TopQuestions),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            max_repair_attempts=max_repair_attempts,
        )
    except (LLMOutputError, RawOutputError) as exc:
        raise QuestionGenerationError(f"Question generation produced unusable output: {exc}") from exc
    except LLMError as exc:
        raise QuestionGenerationError(f"LLM endpoint failed during question generation: {exc}") from exc

    questions = _dedupe(result.questions)
    if not questions:
        raise QuestionGenerationError("Model returned zero usable questions.")
    return questions[:num_questions]
