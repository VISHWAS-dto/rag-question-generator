"""Dependency wiring for the app.

Clients and the InterviewService are built once at startup and stored on
`app.state`; the dependency functions just hand them to routes. This keeps
per-request work minimal and makes overriding trivial in tests
(`app.dependency_overrides`).
"""

from __future__ import annotations

from fastapi import Request
from shared.config import AppSettings, get_app_settings

from app.clients.llm import LLMClient
from app.clients.rag import RAGClient
from app.orchestration.interview_service import InterviewService


def get_settings() -> AppSettings:
    return get_app_settings()


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_rag_client(request: Request) -> RAGClient:
    return request.app.state.rag_client


def get_interview_service(request: Request) -> InterviewService:
    return request.app.state.interview_service
