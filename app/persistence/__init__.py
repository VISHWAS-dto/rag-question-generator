"""Database layer: ORM models, engine/session management, and Alembic wiring.

The app is stateless; all session/interview/report state lives here, in
Postgres in production (SQLite is supported for local `uv run` with no infra).
Connection pooling is configured from settings so the app scales horizontally
without exhausting the database.
"""
