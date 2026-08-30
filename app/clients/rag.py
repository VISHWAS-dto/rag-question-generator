"""RAG client abstraction.

Business logic retrieves context only through `RAGClient.retrieve`. It never
imports a vector store, an embedding model, or `rag_service.*`.

  * `HTTPRAGClient` - production. Calls the standalone `rag` service over HTTP
    with explicit timeouts, bounded retries, and connection pooling. A RAG
    outage surfaces as a typed `RAGUnavailableError`, which the API maps to
    503 - it never crashes the app.
  * `InProcessRAGClient` - single-node / low-footprint mode. Runs retrieval in
    the app's own process by calling `rag_service` directly. Same interface.
  * `FakeRAGClient` - tests. Returns canned chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import httpx
from shared.config import AppSettings, RagMode
from shared.contracts import RetrievedChunk, RetrieveRequest, RetrieveResponse
from shared.logging import get_logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = get_logger("app.rag")


class RAGError(RuntimeError):
    """Base for all RAG client failures."""


class RAGUnavailableError(RAGError):
    """The RAG service is unreachable, timed out, or has no index yet."""


class RAGClient(Protocol):
    def retrieve(
        self, query: str, *, top_k: int, collection: str | None = None
    ) -> list[RetrievedChunk]: ...

    def health(self) -> bool: ...


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into one text block for an LLM prompt.

    Kept here (not in the RAG service) because it is a prompt-construction
    concern of the app, not a retrieval concern.
    """
    blocks = []
    for chunk in chunks:
        section = chunk.metadata.get("section", "Unknown section")
        blocks.append(f"[Section: {section}]\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# HTTP client (production)
# --------------------------------------------------------------------------- #


class HTTPRAGClient:
    def __init__(self, settings: AppSettings, *, client: httpx.Client | None = None) -> None:
        self._base_url = settings.rag_base_url.rstrip("/")
        self._timeout = settings.rag_request_timeout_seconds
        self._connect_timeout = settings.rag_connect_timeout_seconds
        self._max_retries = settings.rag_max_retries
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(self._timeout, connect=self._connect_timeout),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )

    def retrieve(
        self, query: str, *, top_k: int, collection: str | None = None
    ) -> list[RetrievedChunk]:
        req = RetrieveRequest(query=query, top_k=top_k, collection=collection)

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(RAGUnavailableError),
        )
        def _do_request() -> list[RetrievedChunk]:
            try:
                resp = self._client.post(f"{self._base_url}/retrieve", json=req.model_dump())
            except httpx.TimeoutException as exc:
                raise RAGUnavailableError(f"RAG service timed out after {self._timeout}s") from exc
            except httpx.HTTPError as exc:
                raise RAGUnavailableError(f"RAG service unreachable: {exc}") from exc

            if resp.status_code == 503:
                raise RAGUnavailableError(f"RAG service not ready: {resp.text[:300]}")
            if resp.status_code >= 500:
                raise RAGUnavailableError(
                    f"RAG service returned {resp.status_code}: {resp.text[:300]}"
                )
            if resp.status_code >= 400:
                raise RAGError(f"RAG service rejected the request ({resp.status_code}): "
                               f"{resp.text[:300]}")

            try:
                return RetrieveResponse.model_validate(resp.json()).chunks
            except Exception as exc:
                raise RAGError(f"RAG service returned an unparseable response: {exc}") from exc

        return _do_request()

    def health(self) -> bool:
        try:
            return self._client.get(f"{self._base_url}/health", timeout=5.0).status_code < 500
        except httpx.HTTPError:
            return False


# --------------------------------------------------------------------------- #
# In-process client (single node)
# --------------------------------------------------------------------------- #


class InProcessRAGClient:
    """Runs retrieval in the app's own process. Trades independent scaling for
    one fewer service to deploy; keeps the exact same interface so switching is
    a config flag.
    """

    def __init__(self, settings: AppSettings) -> None:
        from shared.config import get_rag_settings

        self._rag_settings = get_rag_settings()
        self._app_settings = settings

    def retrieve(
        self, query: str, *, top_k: int, collection: str | None = None
    ) -> list[RetrievedChunk]:
        from rag_service import service as rag_service

        try:
            chunks, _ = rag_service.retrieve(
                self._rag_settings, query=query, top_k=top_k, collection=collection
            )
        except rag_service.RetrievalUnavailable as exc:
            raise RAGUnavailableError(str(exc)) from exc
        return chunks

    def health(self) -> bool:
        try:
            from rag_service import vectorstore

            vectorstore.get_embedding_model(self._rag_settings)
            return True
        except Exception:
            return False


# --------------------------------------------------------------------------- #
# Fake client (tests)
# --------------------------------------------------------------------------- #


@dataclass
class FakeRAGClient:
    chunks: list[RetrievedChunk] = field(
        default_factory=lambda: [
            RetrievedChunk(
                text="Financial Diligence\n\nReview 12 months of bank statements and "
                "revenue recognition policy.",
                metadata={"section": "Financial Diligence"},
                score=0.9,
            )
        ]
    )
    raise_error: Exception | None = None
    queries_seen: list[str] = field(default_factory=list)

    def retrieve(
        self, query: str, *, top_k: int, collection: str | None = None
    ) -> list[RetrievedChunk]:
        self.queries_seen.append(query)
        if self.raise_error is not None:
            raise self.raise_error
        return self.chunks[:top_k]

    def health(self) -> bool:
        return self.raise_error is None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_rag_client(settings: AppSettings) -> RAGClient:
    if settings.rag_mode == RagMode.HTTP:
        return HTTPRAGClient(settings)
    if settings.rag_mode == RagMode.IN_PROCESS:
        return InProcessRAGClient(settings)
    if settings.rag_mode == RagMode.FAKE:
        return FakeRAGClient()
    raise ValueError(f"Unknown RAG mode: {settings.rag_mode}")
