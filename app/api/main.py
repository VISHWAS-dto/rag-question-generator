"""FastAPI application factory for the `app` service.

Run: uvicorn app.api.main:app --port 8000

Startup builds the LLM and RAG clients and the InterviewService once and puts
them on `app.state`. Nothing is constructed at import time, so importing this
module (e.g. in tests) has no side effects and does not require a database or
any environment variables.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from shared.config import AppSettings, Environment, get_app_settings
from shared.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)

from app.api import health, routes
from app.clients.llm import build_llm_client
from app.clients.rag import build_rag_client
from app.orchestration.interview_service import InterviewService

log = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: AppSettings = get_app_settings()
    configure_logging(
        level=settings.log_level, json_output=settings.log_json, service=settings.service_name
    )
    log.info(
        "app_starting",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        rag_mode=settings.rag_mode,
        db_dialect="sqlite" if settings.is_sqlite else "external",
    )

    # Local/dev convenience: auto-create tables for SQLite so `uv run` works
    # with zero setup. Production uses `alembic upgrade head` and a real DB.
    if settings.is_sqlite and settings.environment in (Environment.LOCAL, Environment.TEST):
        from app.persistence.database import create_all_tables

        create_all_tables(settings)

    app.state.settings = settings
    app.state.llm_client = build_llm_client(settings)
    app.state.rag_client = build_rag_client(settings)
    app.state.interview_service = InterviewService(
        llm=app.state.llm_client, rag=app.state.rag_client, settings=settings
    )
    yield
    log.info("app_stopping")


def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or get_app_settings()
    app = FastAPI(
        title="RAG Due-Diligence API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=settings.cors_origins_list != ["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observability(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        bind_request_context(
            request_id=request_id, path=request.url.path, method=request.method
        )

        # Body-size guard: reject oversized payloads before reading them.
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > settings.max_request_body_bytes:
                    clear_request_context()
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body too large", "request_id": request_id},
                    )
            except ValueError:
                pass

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled_exception")
            clear_request_context()
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info(
            "request_complete", status_code=response.status_code, elapsed_ms=elapsed_ms
        )
        response.headers["x-request-id"] = request_id
        clear_request_context()
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "request_id": request.headers.get("x-request-id")},
        )

    app.include_router(health.router)
    app.include_router(routes.router)
    return app


app = create_app()
