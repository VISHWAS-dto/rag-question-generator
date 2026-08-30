"""Typed errors the orchestration layer raises. The API maps each to an HTTP
status code in one place (see app/api/routes.py).
"""

from __future__ import annotations


class OrchestrationError(RuntimeError):
    """Base class."""


class NotFoundError(OrchestrationError):
    """A referenced session / question / report does not exist. -> 404"""


class ValidationError(OrchestrationError):
    """Caller-supplied data is invalid or the action is not allowed now. -> 422"""


class ConflictError(OrchestrationError):
    """The request conflicts with current state (e.g. already answered). -> 409"""


class IncompleteSessionError(ConflictError):
    """Report requested before every question is answered. -> 409"""


class UpstreamError(OrchestrationError):
    """A dependency (LLM or RAG) failed in a way the caller can't fix. -> 502"""


class DependencyUnavailableError(OrchestrationError):
    """A dependency is down / not ready. -> 503"""


class UpstreamTimeoutError(OrchestrationError):
    """A dependency did not respond in time. -> 504"""
