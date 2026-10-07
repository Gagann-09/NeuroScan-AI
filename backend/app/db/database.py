"""
Database engine, session factory, and session management.
Connection URL is sourced from the centralized config module.
Declarative base is defined in base.py to avoid requiring
a database connection for metadata operations.

Initialization is lazy to allow test monkeypatching of environment variables.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.db.base import Base

_engine = None
_SessionLocal = None


def _get_engine():
    """Lazily create and return the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.DATABASE_URL)
    return _engine


def _get_session_local():
    """Lazily create and return the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal


def get_db():
    db = _get_session_local()()
    try:
        yield db
    finally:
        db.close()


# Backward compatibility: SessionLocal as a callable property
class _SessionLocalProxy:
    def __call__(self):
        return _get_session_local()()


SessionLocal = _SessionLocalProxy()