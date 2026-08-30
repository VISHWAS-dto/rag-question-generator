"""The generate -> parse -> repair loop (app/llm_repair.py).

Ports tests/test_repair_graph.py to the new `LLMClient` seam: a scripted
`FakeLLMClient` returns a malformed first response, then valid output, and we
assert the loop repairs (re-prompts with the error) and eventually returns the
parsed object - or raises once attempts are spent.
"""

from __future__ import annotations

import pytest
from app.clients.llm import FakeLLMClient, LLMUnavailableError, Message
from app.domain.parsing import parse_json_object
from app.llm_repair import LLMOutputError, run_with_repair
from pydantic import BaseModel

pytestmark = pytest.mark.unit


class Widget(BaseModel):
    name: str
    count: int


MESSAGES = [
    Message(role="system", content='Return JSON: {"name": str, "count": int}'),
    Message(role="user", content="Make a widget."),
]


def _parse(raw: str) -> Widget:
    return parse_json_object(raw, Widget)


def _run(llm: FakeLLMClient, **kw):
    return run_with_repair(
        llm, MESSAGES, _parse, max_tokens=256, temperature=0.0, **kw
    )


def test_valid_first_try_no_repair():
    llm = FakeLLMClient(responses=['{"name": "sprocket", "count": 3}'])
    result = _run(llm)
    assert result == Widget(name="sprocket", count=3)
    assert len(llm.prompts_seen) == 1


def test_repairs_after_malformed_json():
    llm = FakeLLMClient(
        responses=["here you go: {name: sprocket, count: 3}", '{"name": "sprocket", "count": 3}']
    )
    result = _run(llm)
    assert result.count == 3
    assert len(llm.prompts_seen) == 2
    # The repair prompt carries the parse error and a corrective instruction.
    assert "not valid JSON" in llm.prompts_seen[1]
    assert "corrected" in llm.prompts_seen[1].lower()
    # ...and the model's own bad output, so it can see what to fix.
    assert "here you go" in llm.prompts_seen[1]


def test_repairs_after_schema_violation():
    llm = FakeLLMClient(
        responses=['{"name": "sprocket"}', '{"name": "sprocket", "count": 7}']
    )
    result = _run(llm)
    assert result.count == 7
    assert "schema" in llm.prompts_seen[1].lower()


def test_raises_after_repair_attempts_exhausted():
    llm = FakeLLMClient(responses=["nope"] * 5)
    with pytest.raises(LLMOutputError, match="did not contain a JSON object"):
        _run(llm, max_repair_attempts=2)
    assert len(llm.prompts_seen) == 3  # 1 initial + 2 repairs


def test_respects_custom_max_repair_attempts():
    llm = FakeLLMClient(responses=["bad", "bad", '{"name": "ok", "count": 1}'])
    result = _run(llm, max_repair_attempts=2)
    assert result.name == "ok"
    assert len(llm.prompts_seen) == 3


def test_endpoint_error_is_not_repaired():
    """An endpoint failure (not a parse failure) propagates immediately - a
    corrective re-prompt cannot fix an unreachable server.
    """

    def boom(_prompt: str) -> str:
        raise LLMUnavailableError("connection refused")

    llm = FakeLLMClient(responses=[boom, '{"name": "x", "count": 1}'])
    with pytest.raises(LLMUnavailableError):
        _run(llm)
    assert len(llm.prompts_seen) == 1
