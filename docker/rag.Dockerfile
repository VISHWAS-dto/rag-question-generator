# --- rag service image --------------------------------------------------------
# Embeddings + vector store + retrieval + ingestion. This is the heavy image
# (torch, sentence-transformers, chromadb). The embedding model is baked in at
# build time so container start does not depend on a model download.
#
# Run from source on PYTHONPATH (see app.Dockerfile for the rationale).

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /build

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra rag

# Pre-download the embedding model into the image so startup is offline.
ENV HF_HOME=/opt/hf-cache
RUN /build/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"


FROM python:3.12-slim AS runtime

RUN groupadd --system rag && useradd --system --gid rag --home /app rag

WORKDIR /app
COPY --from=builder --chown=rag:rag /build/.venv /app/.venv
COPY --from=builder --chown=rag:rag /opt/hf-cache /opt/hf-cache
COPY --chown=rag:rag rag_service /app/rag_service
COPY --chown=rag:rag shared /app/shared
COPY --chown=rag:rag scripts /app/scripts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=8100

USER rag
EXPOSE 8100

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8100/health').status==200 else 1)"

CMD ["uvicorn", "rag_service.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "2"]
