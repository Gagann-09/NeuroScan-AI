"""
Centralized application configuration using Pydantic BaseSettings.
All environment variables are managed here as the single source of truth.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── Database ──
    DATABASE_URL: str = "postgresql://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core"

    # ── MinIO Object Storage ──
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False

    # ── Redis / Celery ──
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Application ──
    APP_TITLE: str = "NeuroScan AI API"
    APP_VERSION: str = "1.0.0"
    CORS_ORIGINS: list[str] = ["*"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton to avoid re-reading .env on every access."""
    return Settings()
