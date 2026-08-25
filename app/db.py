"""SQLite persistence for Phase 2: sessions, questions, answers."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.orm import sessionmaker

from app.config import SQLITE_DB_PATH, SQLITE_URL

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        Path(SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        # timeout: how long SQLite waits on a locked db before raising
        # "database is locked", instead of failing immediately (the default).
        # Needed because a session/answer request can hold a write for the
        # duration of an LLM call, and a second request can land during it.
        _engine = create_engine(
            SQLITE_URL, connect_args={"check_same_thread": False, "timeout": 30}
        )
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def init_db() -> None:
    """Create all tables if they don't already exist."""
    from app.models import Base  # local import to avoid a circular import at module load

    Base.metadata.create_all(get_engine())


def get_db() -> DBSession:
    """Yield-style dependency for FastAPI, and a plain factory for scripts/tests."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()
