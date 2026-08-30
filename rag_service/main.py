"""RAG service FastAPI app.

Run: uvicorn rag_service.main:app --port 8100
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from shared.config import RagSettings, get_rag_settings
from shared.contracts import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ReadinessResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from shared.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)

from rag_service import service, vectorstore

log = get_logger("rag.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_rag_settings()
    configure_logging(
        level=settings.log_level, json_output=settings.log_json, service=settings.service_name
    )
    log.info("rag_starting", environment=settings.environment, chroma_mode=settings.chroma_mode)
    # Warm the embedding model so the first real request isn't slow and a
    # broken model config fails at boot, not mid-request.
    try:
        vectorstore.get_embedding_model(settings)
    except Exception as exc:
        log.error("embedding_model_warmup_failed", error=str(exc))
    yield
    log.info("rag_stopping")


app = FastAPI(title="RAG Service", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    bind_request_context(request_id=request_id, path=request.url.path, method=request.method)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled_exception")
        clear_request_context()
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    log.info("request_complete", status_code=response.status_code, elapsed_ms=elapsed_ms)
    response.headers["x-request-id"] = request_id
    clear_request_context()
    return response


def settings_dep() -> RagSettings:
    return get_rag_settings()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="rag")


@app.get("/ready", response_model=ReadinessResponse)
async def ready(
    response: Response, settings: RagSettings = Depends(settings_dep)
) -> ReadinessResponse:
    checks = service.readiness_checks(settings)
    ok = all(v == "ok" or v.startswith("empty") for v in checks.values())
    if not ok:
        response.status_code = 503
    return ReadinessResponse(status="ok" if ok else "degraded", service="rag", checks=checks)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest, settings: RagSettings = Depends(settings_dep)
) -> RetrieveResponse:
    try:
        chunks, collection = service.retrieve(
            settings, query=body.query, top_k=body.top_k, collection=body.collection
        )
    except service.RetrievalUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return RetrieveResponse(
        chunks=chunks, collection=collection, query_chars=len(body.query)
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    body: IngestRequest, settings: RagSettings = Depends(settings_dep)
) -> IngestResponse:
    try:
        sections, chunks, url, collection = service.ingest(
            settings, source_url=body.source_url, recreate=body.recreate
        )
    except service.IngestionFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return IngestResponse(
        collection=collection,
        sections_extracted=sections,
        chunks_indexed=chunks,
        source_url=url,
    )
