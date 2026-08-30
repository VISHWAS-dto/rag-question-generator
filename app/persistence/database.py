"""Engine + session factory, configured from settings.

One engine per process, created on first use. Pooling parameters come from
`AppSettings` so a horizontally-scaled deployment can size its total
connection count against the database's limit. SQLite (local only) gets the
connection args it needs; Postgres gets a real pool.

`session_scope()` is a context manager for scripts and background tasks.
`get_session()` is the FastAPI dependency (one session per request, always
closed).
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

from shared.config import AppSettings, get_app_settings
from shared.logging import get_logger
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

log = get_logger("app.persistence")

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _build_engine(settings: AppSettings) -> Engine:
    if settings.is_sqlite:
        db_path = settings.database_url.split("///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False, "timeout": 30},
            future=True,
        )
    return create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_seconds,
        pool_recycle=settings.db_pool_recycle_seconds,
        pool_pre_ping=True,
        future=True,
    )


def get_engine(settings: AppSettings | None = None) -> Engine:
    global _engine
    if _engine is None:
        _engine = _build_engine(settings or get_app_settings())
        log.info("db_engine_created", dialect=_engine.dialect.name)
    return _engine


def get_session_factory(settings: AppSettings | None = None) -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(settings), expire_on_commit=False, future=True
        )
    return _session_factory


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency: one DB session per request."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all_tables(settings: AppSettings | None = None) -> None:
    """Create tables directly (used for SQLite local dev and tests where
    running Alembic would be overkill). Production uses `alembic upgrade head`.
    """
    from app.persistence.models import Base

    Base.metadata.create_all(get_engine(settings))


def check_database(settings: AppSettings | None = None) -> bool:
    from sqlalchemy import text

    try:
        with get_engine(settings).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        log.error("db_check_failed", error=str(exc))
        return False


def reset_engine() -> None:
    """Test hook: drop the cached engine/factory."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
