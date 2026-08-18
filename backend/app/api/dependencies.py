"""
Shared FastAPI dependencies.
Centralizes dependency injection so routers never define their own db sessions.
"""
from app.db.database import get_db

__all__ = ["get_db"]
