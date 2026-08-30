"""LLM client abstraction.

The rest of the codebase talks to the model only through `LLMClient.complete`.
Concrete implementations:

  * `OpenAICompatLLMClient` - production. Points at a self-hosted vLLM server
    (or any OpenAI-compatible endpoint: NVIDIA NIM, OpenAI, TGI...). Only the
    base URL, API key, and model name change between them. Sensitive data
    stays inside the deployment because the base URL is an in-house vLLM
    service, never a public API.
  * `EchoLLMClient` - local dev / CI. Returns a deterministic, schema-shaped
    JSON stub with no network call, so the whole stack runs with `uv run` and
    no GPU.
  * `FakeLLMClient` - tests. Returns scripted responses and records prompts.

`complete()` is intentionally synchronous and returns a plain string (the
assistant message content). Retries, timeouts, and connection pooling are the
client's responsibility, not the caller's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from shared.config import AppSettings, LLMProvider
from shared.logging import get_logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = get_logger("app.llm")


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMError(RuntimeError):
    """Base for all LLM client failures."""


class LLMTimeoutError(LLMError):
    """The model did not respond within the configured timeout."""


class LLMUnavailableError(LLMError):
    """The model endpoint is unreachable or returned a server error."""


class LLMBadResponseError(LLMError):
    """The endpoint responded but the payload was not usable."""


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float | None = None,
    ) -> str: ...

    def health(self) -> bool: ...


# --------------------------------------------------------------------------- #
# OpenAI-compatible client (vLLM / NVIDIA NIM / OpenAI / TGI)
# --------------------------------------------------------------------------- #


class OpenAICompatLLMClient:
    def __init__(self, settings: AppSettings, *, client: httpx.Client | None = None) -> None:
        self._base_url = settings.llm_base_url.rstrip("/")
        self._model = settings.llm_model
        self._api_key = settings.llm_api_key
        self._default_temperature = settings.llm_temperature
        self._request_timeout = settings.llm_request_timeout_seconds
        self._connect_timeout = settings.llm_connect_timeout_seconds
        self._max_retries = settings.llm_max_retries
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(
                self._request_timeout, connect=self._connect_timeout
            ),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            headers={"Authorization": f"Bearer {self._api_key}"} if self._api_key else {},
        )

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float | None = None,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        request_timeout = httpx.Timeout(
            timeout or self._request_timeout, connect=self._connect_timeout
        )

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(LLMUnavailableError),
        )
        def _do_request() -> str:
            try:
                resp = self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    timeout=request_timeout,
                )
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(
                    f"LLM request timed out after {request_timeout.read}s"
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMUnavailableError(f"LLM endpoint unreachable: {exc}") from exc

            if resp.status_code >= 500:
                raise LLMUnavailableError(
                    f"LLM endpoint returned {resp.status_code}: {resp.text[:500]}"
                )
            if resp.status_code == 429:
                raise LLMUnavailableError("LLM endpoint rate-limited the request (429)")
            if resp.status_code >= 400:
                raise LLMBadResponseError(
                    f"LLM endpoint returned {resp.status_code}: {resp.text[:500]}"
                )

            try:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                raise LLMBadResponseError(
                    f"LLM response missing choices[0].message.content: {resp.text[:500]}"
                ) from exc

            if not isinstance(content, str) or not content.strip():
                raise LLMBadResponseError("LLM returned empty message content")
            return content

        return _do_request()

    def health(self) -> bool:
        try:
            resp = self._client.get(f"{self._base_url}/models", timeout=5.0)
            return resp.status_code < 500
        except httpx.HTTPError:
            return False


# --------------------------------------------------------------------------- #
# Echo client (local dev / CI, no network)
# --------------------------------------------------------------------------- #


class EchoLLMClient:
    """Returns a deterministic JSON object shaped like whatever the repair
    graph's parser expects, inferred from the system prompt. This is enough to
    exercise the full request path locally without a model.
    """

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float | None = None,
    ) -> str:
        system = next((m.content for m in messages if m.role == "system"), "").lower()
        # Discriminate on phrases unique to each engine's system prompt.
        if "conducting a live, interactive investor questioning session" in system:
            return json.dumps(
                {
                    "follow_up_required": False,
                    "question": None,
                    "category": None,
                    "priority": None,
                    "reason": "ECHO stub: no follow-up needed.",
                }
            )
        if "structured investment assessment from a completed founder interview" in system:
            return json.dumps(_echo_analysis())
        # default: the top-N question generator
        return json.dumps({"questions": [_echo_question(i) for i in range(1, 11)]})

    def health(self) -> bool:
        return True


def _echo_question(i: int) -> dict:
    return {
        "question": f"ECHO stub question {i}: describe an aspect of the business in detail.",
        "category": "Team",
        "priority": "Medium",
        "reason": "ECHO stub reason.",
        "source_context": "ECHO stub section.",
    }


def _echo_analysis() -> dict:
    categories = [
        "market",
        "product",
        "traction",
        "business_model",
        "financials",
        "team",
        "competition",
        "technology",
        "go_to_market",
        "risk",
    ]
    return {
        "executive_summary": "ECHO stub: deterministic placeholder analysis.",
        "strengths": [],
        "risks": [],
        "information_gaps": [],
        "contradictions": [],
        "category_assessments": [
            {
                "category": c,
                "assessment": "Moderate",
                "rationale": "ECHO stub rationale.",
                "evidence_strength": "Low",
                "evidence_gaps": ["ECHO stub gap"],
            }
            for c in categories
        ],
        "recommendations": [],
    }


# --------------------------------------------------------------------------- #
# Fake client (tests)
# --------------------------------------------------------------------------- #


@dataclass
class FakeLLMClient:
    """Scripted LLM for tests. `responses` is a list of strings returned in
    order; a callable receives the rendered prompt text and returns a string.
    Every prompt is recorded in `prompts_seen`.
    """

    responses: list[str | object] = field(default_factory=list)
    prompts_seen: list[str] = field(default_factory=list)
    _idx: int = 0

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        timeout: float | None = None,
    ) -> str:
        rendered = "\n\n".join(f"[{m.role}]\n{m.content}" for m in messages)
        self.prompts_seen.append(rendered)
        if self._idx >= len(self.responses):
            raise LLMBadResponseError("FakeLLMClient ran out of scripted responses")
        item = self.responses[self._idx]
        self._idx += 1
        if callable(item):
            return item(rendered)
        return item  # type: ignore[return-value]

    def health(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_llm_client(settings: AppSettings) -> LLMClient:
    if settings.llm_provider == LLMProvider.OPENAI_COMPAT:
        return OpenAICompatLLMClient(settings)
    if settings.llm_provider == LLMProvider.ECHO:
        return EchoLLMClient()
    if settings.llm_provider == LLMProvider.FAKE:
        return FakeLLMClient()
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
