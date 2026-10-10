"""
Shared FastAPI dependencies.
Centralizes dependency injection so routers never define their own db sessions.
"""
from app.db.database import get_db
from app.core.security import verify_id_token, get_current_user, get_current_user_claims

__all__ = ["get_db", "verify_id_token", "get_current_user", "get_current_user_claims"]
