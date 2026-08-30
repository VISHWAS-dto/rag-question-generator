"""Outbound clients for the two services `app` depends on: the LLM and RAG.

Both are defined as narrow Protocols with multiple implementations (HTTP for
production, in-process / echo for single-node and local dev, fake for tests).
Business logic depends only on the Protocol, never on a concrete provider - so
swapping vLLM for another OpenAI-compatible backend, or the RAG service for an
in-process retriever, is a config change, not a code change.
"""
