"""Retrieval + ingestion orchestration for the RAG service.

Thin layer between the HTTP API and the vector store: applies settings,
enforces bounds, and turns lower-level errors into typed ones the API maps to
HTTP status codes.
"""

from __future__ import annotations

from shared.config import RagSettings
from shared.contracts import RetrievedChunk
from shared.logging import get_logger

from rag_service import vectorstore
from rag_service.chunking import chunk_article
from rag_service.ingest import ExtractionError, FetchError, load_article

log = get_logger("rag.service")


class RetrievalUnavailable(RuntimeError):
    """The vector store cannot serve a query (empty, missing, or backend down)."""


class IngestionFailed(RuntimeError):
    """The source document could not be fetched, parsed, or indexed."""


def retrieve(
    settings: RagSettings, *, query: str, top_k: int, collection: str | None
) -> tuple[list[RetrievedChunk], str]:
    name = collection or settings.chroma_collection
    k = max(1, min(top_k, settings.max_top_k))
    try:
        raw = vectorstore.similarity_search(settings, name, query, k)
    except LookupError as exc:
        raise RetrievalUnavailable(str(exc)) from exc
    except Exception as exc:
        log.error("retrieval_failed", collection=name, error=str(exc))
        raise RetrievalUnavailable(f"Vector store query failed: {exc}") from exc

    return [RetrievedChunk(**c) for c in raw], name


def ingest(
    settings: RagSettings, *, source_url: str | None, recreate: bool
) -> tuple[int, int, str, str]:
    url = source_url or settings.source_url
    name = settings.chroma_collection
    try:
        article = load_article(url, timeout_seconds=settings.fetch_timeout_seconds)
    except (FetchError, ExtractionError) as exc:
        raise IngestionFailed(f"Could not load source article: {exc}") from exc

    chunks = chunk_article(
        article, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    if not recreate and vectorstore.collection_count(settings, name) > 0:
        return len(article.sections), vectorstore.collection_count(settings, name), url, name

    try:
        indexed = vectorstore.rebuild_collection(settings, name, chunks)
    except Exception as exc:
        raise IngestionFailed(f"Failed to embed/index chunks: {exc}") from exc

    log.info("ingest_complete", collection=name, sections=len(article.sections), chunks=indexed)
    return len(article.sections), indexed, url, name


def readiness_checks(settings: RagSettings) -> dict[str, str]:
    checks: dict[str, str] = {}
    try:
        vectorstore.get_embedding_model(settings)
        checks["embedding_model"] = "ok"
    except Exception as exc:
        checks["embedding_model"] = f"error: {exc}"
    try:
        count = vectorstore.collection_count(settings, settings.chroma_collection)
        checks["vector_store"] = "ok" if count > 0 else "empty (run ingestion)"
    except Exception as exc:
        checks["vector_store"] = f"error: {exc}"
    return checks
