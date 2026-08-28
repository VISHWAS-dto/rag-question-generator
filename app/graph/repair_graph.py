"""A reusable generate -> parse -> repair StateGraph.

Every LLM engine in this project prompts the hosted NVIDIA model for a single
raw-JSON object and then parses + Pydantic-validates it by hand (the model
does not reliably support tool-calling / `with_structured_output`). Before
LangGraph, a malformed or schema-invalid response just raised, surfacing as
an HTTP 500.

This graph keeps that exact parse logic but wraps it in a loop:

    generate --> parse --(ok)------------> END
                   ^                        |
                   |                        v
                 repair <--(fail, attempts left)

On a parse failure the `repair` node re-invokes the model with the validation
error appended to the original prompt, asking for corrected JSON only. After
`max_repair_attempts` exhausted re-invocations it raises the *last* parse
error, so callers see the same RuntimeError type they did before.

The graph is generic over:
  * the LLM (anything exposing `.invoke(prompt_value) -> obj_with_.content`),
  * the prompt (a `ChatPromptTemplate`) plus its input dict,
  * a `parser(raw_text) -> ParsedT` callable that raises on bad input.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Optional, TypeVar

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.config import LLM_MAX_REPAIR_ATTEMPTS

ParsedT = TypeVar("ParsedT")


class SupportsInvoke:
    """Structural type only, for documentation — anything with a compatible
    `.invoke()` (real ChatNVIDIA runnable or a test stub) is accepted.
    """

    def invoke(self, prompt_value: Any) -> Any: ...  # pragma: no cover


class RepairState(TypedDict, total=False):
    # Immutable inputs, set once by the caller.
    prompt_inputs: dict
    # The rendered base prompt (ChatPromptValue) reused every attempt.
    base_prompt_value: Any
    # Working fields, updated as the graph runs.
    raw_text: str
    parse_error: Optional[str]
    attempts: int  # number of repair re-invocations done so far
    parsed: Any  # the validated object once parsing succeeds


def _augmented_prompt_value(base: Any, parse_error: str) -> Any:
    """Return a new prompt value = base messages + a corrective instruction.

    `base` is a ChatPromptValue; we append one HumanMessage naming the error
    and asking for corrected JSON. Falls back to wrapping in a fresh
    ChatPromptValue if `base` isn't the expected type (keeps test stubs that
    only read `.to_string()` / iterate messages working).
    """
    correction = HumanMessage(
        content=(
            "Your previous response could not be used. It failed with this "
            f"error:\n{parse_error}\n\n"
            "Return a corrected response that is ONLY a single valid JSON "
            "object matching the schema described above — no markdown code "
            "fences, no prose before or after."
        )
    )
    try:
        messages = list(base.to_messages())
    except AttributeError:
        messages = list(getattr(base, "messages", []))
    return ChatPromptValue(messages=[*messages, correction])


def build_repair_graph(
    llm: Any,
    prompt: ChatPromptTemplate,
    parser: Callable[[str], ParsedT],
    *,
    max_repair_attempts: int = LLM_MAX_REPAIR_ATTEMPTS,
):
    """Compile and return a StateGraph that runs generate -> parse -> repair.

    Invoke it with ``{"prompt_inputs": {...}}``; read ``result["parsed"]`` for
    the validated object. Raises the last parse error (a RuntimeError, matching
    the pre-LangGraph behaviour) if every repair attempt is exhausted.
    """

    def generate(state: RepairState) -> dict:
        prompt_value = prompt.invoke(state["prompt_inputs"])
        response = llm.invoke(prompt_value)
        return {
            "base_prompt_value": prompt_value,
            "raw_text": response.content,
            "attempts": 0,
            "parse_error": None,
        }

    def parse(state: RepairState) -> dict:
        try:
            parsed = parser(state["raw_text"])
        except Exception as exc:  # noqa: BLE001 - re-raised below if attempts exhausted
            return {"parse_error": str(exc), "parsed": None}
        return {"parsed": parsed, "parse_error": None}

    def repair(state: RepairState) -> dict:
        prompt_value = _augmented_prompt_value(
            state["base_prompt_value"], state.get("parse_error") or "unknown error"
        )
        response = llm.invoke(prompt_value)
        return {
            "raw_text": response.content,
            "attempts": state.get("attempts", 0) + 1,
        }

    def after_parse(state: RepairState) -> str:
        if state.get("parsed") is not None:
            return "ok"
        if state.get("attempts", 0) >= max_repair_attempts:
            # Exhausted — raise the same error type the manual parser raised.
            raise RuntimeError(state.get("parse_error") or "LLM output parsing failed")
        return "repair"

    graph = StateGraph(RepairState)
    graph.add_node("generate", generate)
    graph.add_node("parse", parse)
    graph.add_node("repair", repair)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "parse")
    graph.add_conditional_edges("parse", after_parse, {"ok": END, "repair": "repair"})
    graph.add_edge("repair", "parse")

    return graph.compile()


def run_repair_graph(
    llm: Any,
    prompt: ChatPromptTemplate,
    prompt_inputs: dict,
    parser: Callable[[str], ParsedT],
    *,
    max_repair_attempts: int = LLM_MAX_REPAIR_ATTEMPTS,
) -> ParsedT:
    """Convenience wrapper: build the graph, invoke it once, return the parsed
    object. Engines call this instead of composing `_prompt | llm` by hand.
    """
    compiled = build_repair_graph(
        llm, prompt, parser, max_repair_attempts=max_repair_attempts
    )
    result = compiled.invoke({"prompt_inputs": prompt_inputs})
    return result["parsed"]
