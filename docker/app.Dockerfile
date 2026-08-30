# --- app service image ---------------------------------------------------------
# The API + orchestration + persistence. Intentionally does NOT install the
# heavy `rag` extra (torch, sentence-transformers, chromadb): retrieval is a
# separate service. Multi-stage, uv-based, non-root, with a HEALTHCHECK.
#
# The service is run from source on PYTHONPATH (standard for an application, as
# opposed to a distributed library) so the image needs no project wheel build.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

# Lockfile-driven, reproducible install of just the dependencies (cached across
# code-only changes). --no-install-project: we ship source, not a wheel.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra app


FROM python:3.12-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
COPY --from=builder --chown=app:app /build/.venv /app/.venv

# Application source. `rag_service` is included as pure Python (its `rag` extra
# is NOT installed) so `InProcessRAGClient` can import it for single-node mode;
# in production the app runs APP_RAG_MODE=http and never loads that path.
COPY --chown=app:app app /app/app
COPY --chown=app:app shared /app/shared
COPY --chown=app:app rag_service /app/rag_service
COPY --chown=app:app migrations /app/migrations
COPY --chown=app:app scripts /app/scripts
COPY --chown=app:app alembic.ini /app/alembic.ini

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Sync workers (LLM calls are blocking I/O). Scale with replicas, not threads.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
