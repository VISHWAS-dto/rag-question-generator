"""Wire contracts shared between services.

These Pydantic models are the *only* coupling between `app` and `rag`: the
`rag` service validates requests and serialises responses with them, and the
app's `HTTPRAGClient` uses the same types to build requests and parse
responses. Keeping them here (not in either service) makes the boundary
explicit and lets a contract test import both sides.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=10, ge=1, le=50)
    collection: str | None = Field(
        default=None, description="Override the server's default collection."
    )


class RetrievedChunk(BaseModel):
    text: str
    metadata: dict[str, str | int | float | None] = Field(default_factory=dict)
    score: float | None = Field(
        default=None, description="Similarity score if the backend provides one."
    )


class RetrieveResponse(BaseModel):
    chunks: list[RetrievedChunk]
    collection: str
    query_chars: int


class IngestRequest(BaseModel):
    source_url: str | None = Field(
        default=None, description="Override the server's configured source URL."
    )
    recreate: bool = Field(
        default=True, description="Drop and rebuild the collection from scratch."
    )


class IngestResponse(BaseModel):
    collection: str
    sections_extracted: int
    chunks_indexed: int
    source_url: str


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadinessResponse(BaseModel):
    status: str
    service: str
    checks: dict[str, str]
