# RAG Due-Diligence Engine

Generates the top-N due-diligence questions for a startup, runs an interactive
follow-up interview, and produces a scored, evidence-grounded investor report.

The AI only produces qualitative findings in words. Every numeric score and the
overall risk level are computed by fixed Python rules, so they are consistent
and auditable.

## Architecture

Three independently deployable services plus shared infrastructure. They
communicate over HTTP; the same container images run in local, staging, and
production — only environment variables differ.

```
frontend (nginx)  ──/api──►  app (FastAPI, stateless, N replicas)
                                 │                    │
                             HTTP │                HTTP │            SQL (pooled)
                                 ▼                    ▼                 ▼
                          rag service          llm service         PostgreSQL
                          (FastAPI, N)         (vLLM, OpenAI-       (sessions,
                          • /retrieve           compatible,         questions,
                          • /ingest             self-hosted,        answers,
                          • embeddings          in-house model)     reports)
                          • vector store        NO public API
```

| Service | Responsibility | Scaling axis |
|---------|----------------|--------------|
| **app** | REST API, request validation, interview state machine, deterministic scoring, persistence | I/O-bound → add replicas |
| **rag** | Fetch + chunk the knowledge base, embeddings (MiniLM), vector store (Chroma), similarity search | CPU/memory-bound → own replicas |
| **llm** | Self-hosted model inference, OpenAI-compatible API (vLLM) | GPU-bound → own cluster |
| **postgres** | All session/interview/report state | pooling now; read replicas later |

**How the app talks to RAG** — a `RAGClient` protocol (`app/clients/rag.py`).
`HTTPRAGClient` (prod) calls the `rag` service with timeouts, bounded retries,
and a typed `RAGUnavailableError` on failure. `InProcessRAGClient` keeps
retrieval in-process for single-node mode. `FakeRAGClient` is used in tests.
No engine imports a vector store or `rag_service.*` directly.

**How the app talks to the LLM** — an `LLMClient` protocol
(`app/clients/llm.py`). `OpenAICompatLLMClient` points at self-hosted vLLM (or
any OpenAI-compatible endpoint) via `APP_LLM_BASE_URL` / `APP_LLM_MODEL` only.
`EchoLLMClient` is a dependency-free stub for local dev and CI. Provider is a
config switch, not a code change. Sensitive founder data never reaches a public
API in production — the base URL is the internal vLLM service.

**Loose coupling** — retrieval, vector storage, LLM inference, and application
logic are separate modules behind narrow interfaces. RAG and the LLM can be
deployed and scaled without touching the app.

## Run locally with UV

```bash
make install                 # uv sync --extra app --extra rag --extra dev
cp .env.example .env          # defaults: echo LLM, in-process RAG, SQLite
make ingest                   # build the local vector index (downloads MiniLM once)

# Option A — one-shot CLI
make questions INFO="We're a B2B SaaS startup, 15 staff, 2 crore INR ARR, seed stage."

# Option B — the API (RAG in-process, echo LLM, SQLite, zero infra)
make run-app                  # http://localhost:8000/docs

# Option C — app + a separately-running RAG service
make run-rag                  # terminal 1
APP_RAG_MODE=http APP_RAG_BASE_URL=http://localhost:8100 make run-app   # terminal 2
```

Tests and checks:

```bash
make test         # unit + integration (no live LLM)
make lint         # ruff + mypy
uv run pytest -m live      # opt-in; needs a real LLM endpoint + RUN_LIVE_TESTS=1
```

## Run with Docker

```bash
make up                       # app + rag + postgres + frontend (echo LLM, no GPU)
make up-ingest                # build the knowledge index inside the rag container
# frontend:  http://localhost:3000
# app docs:  http://localhost:8000/docs
make down
```

Compose layout: `docker-compose.yml` (base, environment-agnostic) +
`docker-compose.override.yml` (local hot-reload, auto-loaded) +
`docker-compose.prod.yml` (adds self-hosted vLLM, replica counts, resource
limits). Scale a service: `docker compose up --scale app=4 --scale rag=2`.

### Production with a self-hosted LLM (vLLM)

```bash
export APP_CORS_ALLOW_ORIGINS=https://your-frontend.example
export VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct   # your in-house model
make prod-config              # validate the merged config
make prod-up                  # needs an NVIDIA GPU host
```

`docker-compose.prod.yml` runs `vllm/vllm-openai` and points the app at
`http://vllm:8000/v1`. The app is unaware it is vLLM — swapping in another
OpenAI-compatible backend is an env change.

## API

```
POST /sessions                    create a session, seed the top-N questions
                                  (send Idempotency-Key to make retries safe)
GET  /sessions/{id}               session status
GET  /sessions/{id}/questions     all questions + the current one to answer
POST /questions/{id}/answer       submit an answer → follow-up / next / complete
POST /sessions/{id}/complete      generate the report (idempotent)
GET  /sessions/{id}/report        fetch the stored report
GET  /health                      liveness
GET  /ready                       readiness (DB + rag + llm), 503 when degraded
```

## Project layout

```
app/
  api/            FastAPI routes, schemas, health checks, app factory
  orchestration/  interview state machine, report assembly, typed errors
  engines/        LLM steps: question generation, follow-up, analysis + prompts
  clients/        LLMClient / RAGClient protocols + HTTP/in-process/echo/fake impls
  domain/         pure schemas, deterministic scoring, risk backstops, JSON parsing
  persistence/    ORM models, engine/session factory (pooled)
  llm_repair.py   generate → parse → repair loop around raw-JSON LLM output
rag_service/      standalone RAG service (ingest, chunk, embed, retrieve)
shared/           config (pydantic-settings), structured logging, wire contracts
migrations/       Alembic
docker/           per-service Dockerfiles, nginx config
tests/            unit / integration / load (locust)
```

## Before production deployment

- Rotate the NVIDIA API key that a previous commit placed in `.env` (it is no
  longer used — production is self-hosted — but it was exposed).
- Provision a managed Postgres and set `APP_DATABASE_URL`; run `alembic upgrade head`.
- Stand up the vLLM host (GPU) and load the in-house model.
- Set `APP_CORS_ALLOW_ORIGINS` to the real frontend origin(s).
- Put a TLS-terminating load balancer in front of `frontend` / `app`.
- Add authentication (there is a clean seam at the API dependency layer).
- Point `rag` at a standalone Chroma (`RAG_CHROMA_MODE=http`) so replicas share state.
- Wire logs/metrics to your aggregator (logs are already structured JSON with request IDs).
- Run a load test (`tests/load/locustfile.py`) against a staging stack.
