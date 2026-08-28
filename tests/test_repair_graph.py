"""Tests for the generate -> parse -> repair LangGraph (app/graph/repair_graph.py).

These exercise the loop directly with a scripted LLM stub: first response is
deliberately malformed, later responses are valid, and we assert the graph
repairs (re-invokes with the error appended) and eventually returns the parsed
object — or raises the underlying parse error once repair attempts are spent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pytest
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from app.graph.repair_graph import run_repair_graph


class StubResponse:
    def __init__(self, content: str):
        self.content = content


class ScriptedLLM:
    """Returns the next canned string per `.invoke()` call, recording every
    prompt it was given (rendered to text) so tests can assert the repair
    node actually appended the error.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts_seen: list[str] = []

    def invoke(self, prompt_value) -> StubResponse:
        self.prompts_seen.append(prompt_value.to_string())
        return StubResponse(self._responses.pop(0))


class Widget(BaseModel):
    name: str
    count: int


_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Return JSON: {{\"name\": str, \"count\": int}}"),
        ("user", "Make a widget called {widget_name}. Respond with ONLY the JSON object."),
    ]
)


def _parse_widget(raw_text: str) -> Widget:
    text = raw_text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"not valid JSON: {exc}") from exc
    try:
        return Widget.model_validate(data)
    except ValidationError as exc:
        raise RuntimeError(f"schema mismatch: {exc}") from exc


def test_valid_first_try_no_repair():
    llm = ScriptedLLM(['{"name": "sprocket", "count": 3}'])
    result = run_repair_graph(llm, _PROMPT, {"widget_name": "sprocket"}, _parse_widget)
    assert isinstance(result, Widget)
    assert result.name == "sprocket" and result.count == 3
    assert len(llm.prompts_seen) == 1  # no repair invocation


def test_repairs_after_malformed_json():
    llm = ScriptedLLM(
        [
            "here you go: {name: sprocket, count: 3",  # not JSON
            '{"name": "sprocket", "count": 3}',  # fixed
        ]
    )
    result = run_repair_graph(llm, _PROMPT, {"widget_name": "sprocket"}, _parse_widget)
    assert result.count == 3
    assert len(llm.prompts_seen) == 2
    # The repair prompt must carry the parse error and a corrective instruction.
    assert "not valid JSON" in llm.prompts_seen[1]
    assert "corrected" in llm.prompts_seen[1].lower()


def test_repairs_after_schema_violation():
    llm = ScriptedLLM(
        [
            '{"name": "sprocket"}',  # missing required "count"
            '{"name": "sprocket", "count": 7}',  # fixed
        ]
    )
    result = run_repair_graph(llm, _PROMPT, {"widget_name": "sprocket"}, _parse_widget)
    assert result.count == 7
    assert "schema mismatch" in llm.prompts_seen[1]


def test_raises_underlying_error_when_repair_attempts_exhausted():
    llm = ScriptedLLM(["nope"] * 5)  # every attempt malformed
    with pytest.raises(RuntimeError, match="not valid JSON"):
        run_repair_graph(
            llm, _PROMPT, {"widget_name": "x"}, _parse_widget, max_repair_attempts=2
        )
    # 1 initial + 2 repair invocations, then give up.
    assert len(llm.prompts_seen) == 3


def test_respects_custom_max_repair_attempts():
    llm = ScriptedLLM(["bad", "bad", '{"name": "ok", "count": 1}'])
    result = run_repair_graph(
        llm, _PROMPT, {"widget_name": "x"}, _parse_widget, max_repair_attempts=2
    )
    assert result.name == "ok"
    assert len(llm.prompts_seen) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
