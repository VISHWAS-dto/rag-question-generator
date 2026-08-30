"""Generate -> parse -> repair loop around a raw-JSON LLM response.

The self-hosted models this project targets are prompted for a single JSON
object and the response is parsed + validated by hand (tool-calling / native
structured output is not reliably available across backends). A malformed or
schema-invalid response would otherwise fail the whole request.

This loop keeps that parse logic but retries on failure: on a parse error it
re-invokes the model with the error appended and asks for corrected JSON,
up to `max_repair_attempts` times, then raises the last parse error.

This is the same contract the previous LangGraph-based implementation had
(`run_repair_graph`), reimplemented as a plain function so the `app` service
does not depend on langgraph/langchain. The LLM is used only through the
`LLMClient.complete` seam, so a `FakeLLMClient` stands in for tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from shared.logging import get_logger

from app.clients.llm import LLMClient, LLMError, Message

log = get_logger("app.llm_repair")

ParsedT = TypeVar("ParsedT")


class LLMOutputError(RuntimeError):
    """The model's output could not be parsed/validated after all repair attempts."""


_CORRECTION_TEMPLATE = (
    "Your previous response could not be used. It failed with this error:\n{error}\n\n"
    "Return a corrected response that is ONLY a single valid JSON object matching the "
    "schema described above - no markdown code fences, no prose before or after."
)


def run_with_repair(
    llm: LLMClient,
    messages: list[Message],
    parser: Callable[[str], ParsedT],
    *,
    max_tokens: int,
    temperature: float,
    timeout: float | None = None,
    max_repair_attempts: int = 2,
) -> ParsedT:
    """Invoke `llm`, parse the response, and repair-retry on failure.

    Raises `LLMOutputError` if every attempt fails to parse, or the underlying
    `LLMError` if the model endpoint itself is unavailable (that is not
    repairable by re-prompting).
    """
    conversation = list(messages)
    last_error: str | None = None

    for attempt in range(max_repair_attempts + 1):
        try:
            raw = llm.complete(
                conversation,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except LLMError:
            # Endpoint problem - not something a corrective re-prompt fixes.
            raise

        try:
            return parser(raw)
        except Exception as exc:
            last_error = str(exc)
            log.warning(
                "llm_output_parse_failed",
                attempt=attempt,
                max_repair_attempts=max_repair_attempts,
                error=last_error[:500],
            )
            if attempt >= max_repair_attempts:
                break
            conversation = [
                *conversation,
                Message(role="assistant", content=raw),
                Message(
                    role="user",
                    content=_CORRECTION_TEMPLATE.format(error=last_error),
                ),
            ]

    raise LLMOutputError(last_error or "LLM output parsing failed")
