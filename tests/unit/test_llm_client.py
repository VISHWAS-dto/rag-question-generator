"""LLMClient implementations (app/clients/llm.py).

The OpenAI-compatible client is tested against an injected httpx.MockTransport
(no global patching, no version-fragile mock library), covering timeout,
5xx-retry, 4xx, and malformed-payload handling without a real model. This is
the 'test the LLM integration through an abstraction' requirement: engines
depend on `LLMClient`, and this file pins the wire behaviour of the real
implementation.
"""

from __future__ import annotations

import httpx
import pytest
from app.clients.llm import (
    EchoLLMClient,
    LLMBadResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
    OpenAICompatLLMClient,
)
from shared.config import AppSettings

pytestmark = pytest.mark.unit

BASE = "http://llm.test/v1"


def _ok_body(content: str = '{"ok": true}') -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _client(handler, **overrides) -> OpenAICompatLLMClient:
    settings = AppSettings(
        llm_base_url=BASE,
        llm_api_key="k",
        llm_model="test-model",
        llm_max_retries=2,
        llm_request_timeout_seconds=5.0,
        **overrides,
    )
    transport = httpx.MockTransport(handler)
    inner = httpx.Client(transport=transport, headers={"Authorization": "Bearer k"})
    return OpenAICompatLLMClient(settings, client=inner)


def _complete(client: OpenAICompatLLMClient) -> str:
    return client.complete(
        [Message(role="user", content="hi")], max_tokens=10, temperature=0.0
    )


def test_successful_completion():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=_ok_body("hello"))

    assert _complete(_client(handler)) == "hello"


def test_retries_on_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=_ok_body("recovered"))

    assert _complete(_client(handler)) == "recovered"
    assert calls["n"] == 2


def test_5xx_exhausts_retries_and_raises_unavailable():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMUnavailableError):
        _complete(_client(handler))
    assert calls["n"] == 3  # 1 + 2 retries


def test_429_is_retried_as_unavailable():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, text="slow down")

    with pytest.raises(LLMUnavailableError):
        _complete(_client(handler))
    assert calls["n"] == 3


def test_4xx_raises_bad_response_without_retry():
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(LLMBadResponseError):
        _complete(_client(handler))
    assert calls["n"] == 1


def test_timeout_raises_llm_timeout_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMTimeoutError):
        _complete(_client(handler))


def test_connection_error_raises_unavailable():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMUnavailableError):
        _complete(_client(handler))


def test_malformed_payload_raises_bad_response():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    with pytest.raises(LLMBadResponseError):
        _complete(_client(handler))


def test_empty_content_raises_bad_response():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body("   "))

    with pytest.raises(LLMBadResponseError):
        _complete(_client(handler))


def test_echo_client_shapes_by_prompt():
    from app.engines.prompts import (
        FOLLOWUP_SYSTEM_PROMPT,
        analysis_system_prompt,
        question_system_prompt,
    )

    echo = EchoLLMClient()
    q = echo.complete(
        [Message(role="system", content=question_system_prompt(10))],
        max_tokens=10,
        temperature=0.0,
    )
    assert '"questions"' in q
    f = echo.complete(
        [Message(role="system", content=FOLLOWUP_SYSTEM_PROMPT)],
        max_tokens=10,
        temperature=0.0,
    )
    assert '"follow_up_required"' in f
    a = echo.complete(
        [Message(role="system", content=analysis_system_prompt())],
        max_tokens=10,
        temperature=0.0,
    )
    assert '"category_assessments"' in a
