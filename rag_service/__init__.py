"""Standalone RAG service.

Owns everything retrieval: fetching and chunking the source knowledge base,
the embedding model, the vector store, and similarity search. Exposes a small
HTTP API (`/retrieve`, `/ingest`, `/health`, `/ready`) and nothing else. It
has no knowledge of sessions, questions, reports, or the LLM - the `app`
service orchestrates those and calls here only to retrieve context.

This service can be deployed and scaled independently of `app`: it is
CPU/memory-bound (the embedding model), while `app` is I/O-bound.
"""
