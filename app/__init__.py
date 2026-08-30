"""The `app` service: HTTP API, interview orchestration, persistence, and the
outbound LLM/RAG clients.

Layering (each layer may import only from those below it):

    api/           FastAPI routes, request/response schemas, health checks
    orchestration/ interview state machine, report assembly, typed errors
    engines/       LLM-driven steps (question gen, follow-up, analysis)
    clients/       LLMClient / RAGClient protocols + implementations
    domain/        pure schemas, deterministic scoring, risk backstops
    persistence/   ORM models, engine/session factory

Nothing in `app` imports `rag_service` except `clients.rag.InProcessRAGClient`,
which is only used when RAG runs in-process (RAG_MODE=in_process).
"""
