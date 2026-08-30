"""Pure domain layer: schemas, deterministic scoring, and risk logic.

Nothing here performs I/O - no database, no HTTP, no LLM. This is the part of
the system that must be correct and is cheap to test in isolation.
"""
