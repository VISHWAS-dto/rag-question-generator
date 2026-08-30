"""Embedding model + Chroma vector store.

Ported from app/rag/vectorstore.py, generalised so the store can be either an
embedded persistent directory (single node) or a standalone Chroma server
(shared across `rag` replicas), selected by RAG_CHROMA_MODE.

The embedding model is loaded lazily and once per process. This is the
heaviest resource in the whole system (torch + a ~90MB model), which is
exactly why it lives in this service and not in `app`.
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Any

# chromadb 0.5.x + some posthog versions spam telemetry errors to stderr on
# every call ("capture() takes 1 positional argument but 3 were given").
# Harmless but noisy; disable before chromadb is imported, and silence the
# posthog logger as a belt-and-braces fallback for versions that ignore the
# env var.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.product.noop.NoopTelemetryClient")

import logging as _logging

_logging.getLogger("chromadb.telemetry").setLevel(_logging.CRITICAL)
_logging.getLogger("posthog").setLevel(_logging.CRITICAL)

import chromadb
from chromadb.config import Settings as ChromaClientSettings
from sentence_transformers import SentenceTransformer
from shared.config import RagSettings
from shared.logging import get_logger

log = get_logger("rag.vectorstore")

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()
_client: chromadb.ClientAPI | None = None
_client_lock = threading.Lock()


def get_embedding_model(settings: RagSettings) -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                log.info("loading_embedding_model", model=settings.embedding_model_name)
                _model = SentenceTransformer(settings.embedding_model_name)
    return _model


def embed_texts(settings: RagSettings, texts: list[str]) -> list[list[float]]:
    model = get_embedding_model(settings)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()


def get_chroma_client(settings: RagSettings) -> chromadb.ClientAPI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if settings.chroma_mode == "http":
                    log.info(
                        "connecting_chroma_http",
                        host=settings.chroma_host,
                        port=settings.chroma_port,
                    )
                    _client = chromadb.HttpClient(
                        host=settings.chroma_host,
                        port=settings.chroma_port,
                        settings=ChromaClientSettings(anonymized_telemetry=False),
                    )
                else:
                    log.info("opening_chroma_embedded", path=settings.chroma_persist_dir)
                    _client = chromadb.PersistentClient(
                        path=settings.chroma_persist_dir,
                        settings=ChromaClientSettings(anonymized_telemetry=False),
                    )
    return _client


def _collection(settings: RagSettings, name: str) -> chromadb.Collection:
    client = get_chroma_client(settings)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def rebuild_collection(
    settings: RagSettings, name: str, chunks: list[dict[str, Any]]
) -> int:
    """Drop and repopulate the named collection from `chunks`. Returns count."""
    client = get_chroma_client(settings)
    with contextlib.suppress(Exception):  # collection may not exist yet; that's fine
        client.delete_collection(name)

    collection = _collection(settings, name)
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(settings, texts)
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,  # type: ignore[arg-type]  # chromadb accepts list[list[float]] at runtime
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


def similarity_search(
    settings: RagSettings, name: str, query: str, k: int
) -> list[dict[str, Any]]:
    """Return up to `k` nearest chunks as {text, metadata, score} dicts.

    Raises if the collection is missing or empty so the caller can return a
    clear error rather than an empty result that looks like "nothing relevant".
    """
    collection = _collection(settings, name)
    count = collection.count()
    if count == 0:
        raise LookupError(
            f"Vector collection '{name}' is empty. Run ingestion (POST /ingest) first."
        )

    query_embedding = embed_texts(settings, [query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],  # type: ignore[arg-type]
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],  # type: ignore[list-item]
    )

    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    chunks: list[dict[str, Any]] = []
    for i, text in enumerate(docs):
        distance = dists[i] if i < len(dists) else None
        chunks.append(
            {
                "text": text,
                "metadata": metas[i] if i < len(metas) else {},
                # cosine distance -> similarity
                "score": (1.0 - distance) if distance is not None else None,
            }
        )
    return chunks


def collection_count(settings: RagSettings, name: str) -> int:
    return _collection(settings, name).count()


def reset_caches() -> None:
    """Test hook."""
    global _model, _client
    _model = None
    _client = None
