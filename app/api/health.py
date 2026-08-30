"""Liveness and readiness endpoints.

  * GET /health  - liveness: the process is up. Cheap, no dependency checks.
  * GET /ready   - readiness: the process can serve traffic. Checks the DB,
    and (if APP_READINESS_CHECK_DEPENDENCIES) the LLM and RAG services.
    Returns 503 when any hard dependency is down so an orchestrator can hold
    traffic off this replica.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from shared.config import AppSettings

from app.api.dependencies import get_llm_client, get_rag_client, get_settings
from app.clients.llm import LLMClient
from app.clients.rag import RAGClient
from app.persistence.database import check_database

router = APIRouter(tags=["ops"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "app"}


@router.get("/ready")
def ready(
    response: Response,
    settings: AppSettings = Depends(get_settings),
    llm: LLMClient = Depends(get_llm_client),
    rag: RAGClient = Depends(get_rag_client),
) -> dict[str, object]:
    checks: dict[str, str] = {}

    checks["database"] = "ok" if check_database(settings) else "error"

    if settings.readiness_check_dependencies:
        checks["llm"] = "ok" if _safe(llm.health) else "error"
        checks["rag"] = "ok" if _safe(rag.health) else "error"

    ok = all(v == "ok" for v in checks.values())
    if not ok:
        response.status_code = 503
    return {"status": "ok" if ok else "degraded", "service": "app", "checks": checks}


def _safe(fn) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False
