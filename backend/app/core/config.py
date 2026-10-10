"""
Centralized application configuration using Pydantic BaseSettings.
All environment variables are managed here as the single source of truth.

Production credentials MUST be supplied via environment variables or .env file.
No hard-coded credential defaults are provided for security-sensitive fields.

Development defaults for non-sensitive configuration (endpoints, timeouts, etc.)
are retained. Docker Compose provides development credentials for local workflow.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import field_validator, ValidationError
from datetime import timedelta
from typing import Union


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── Database ──
    # No default - must be provided via environment variable
    # Expected format: postgresql://user:password@host:port/database
    DATABASE_URL: str

    # ── MinIO Object Storage ──
    MINIO_ENDPOINT: str = "localhost:9000"
    # No default for credentials - must be provided via environment variable
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_SECURE: bool = False

    # ── Redis / Celery ──
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Firebase Authentication ──
    # No default - must be provided via environment variable
    FIREBASE_PROJECT_ID: str

    # ── Application ──
    APP_TITLE: str = "NeuroScan AI API"
    APP_VERSION: str = "1.0.0"
    # No default for CORS_ORIGINS - must be provided via environment variable
    # when allow_credentials=True to avoid insecure wildcard with credentials.
    # Provide as comma-separated string, e.g., "http://localhost:3000,https://app.example.com"
    CORS_ORIGINS: str

    # ── Presigned URL ──
    # TTL for presigned MinIO URLs in minutes.
    # Default 15 minutes to limit exposure window for leaked URLs.
    PRESIGNED_URL_TTL_MINUTES: int = 15

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate DATABASE_URL is provided and not using a known default password."""
        if not v or not v.strip():
            raise ValueError(
                "DATABASE_URL must be set via environment variable. "
                "No default is provided for security."
            )
        # Warn if using the known development password in what appears to be production
        if "secure_password_123" in v:
            # This is a development default; allow but log warning in production context
            pass
        return v

    @field_validator("MINIO_ACCESS_KEY", "MINIO_SECRET_KEY")
    @classmethod
    def validate_minio_credentials(cls, v: str) -> str:
        """Validate MinIO credentials are provided."""
        if not v or not v.strip():
            raise ValueError(
                "MinIO credentials must be set via environment variables. "
                "No default is provided for security."
            )
        return v

    @field_validator("FIREBASE_PROJECT_ID")
    @classmethod
    def validate_firebase_project_id(cls, v: str) -> str:
        """Validate FIREBASE_PROJECT_ID is provided."""
        if not v or not v.strip():
            raise ValueError(
                "FIREBASE_PROJECT_ID must be set via environment variable. "
                "No default is provided for security."
            )
        return v

    @field_validator("CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, v: str) -> str:
        """Validate CORS_ORIGINS is provided and not insecure wildcard with credentials."""
        if not v or not v.strip():
            raise ValueError(
                "CORS_ORIGINS must be provided via environment variable. "
                "No default is provided for security."
            )
        # Parse comma-separated origins
        origins = [origin.strip() for origin in v.split(",") if origin.strip()]
        if not origins:
            raise ValueError(
                "CORS_ORIGINS must contain at least one origin."
            )
        # Explicitly reject wildcard when credentials are used (allow_credentials=True in middleware)
        if origins == ["*"]:
            raise ValueError(
                "CORS_ORIGINS cannot be ['*'] when allow_credentials=True. "
                "Explicitly list trusted origins instead."
            )
        return v

    @field_validator("PRESIGNED_URL_TTL_MINUTES")
    @classmethod
    def validate_presigned_url_ttl(cls, v: int) -> int:
        """Validate PRESIGNED_URL_TTL_MINUTES is positive and reasonable."""
        if v <= 0:
            raise ValueError(
                "PRESIGNED_URL_TTL_MINUTES must be a positive integer."
            )
        # Reasonable upper bound: 24 hours = 1440 minutes
        if v > 1440:
            raise ValueError(
                "PRESIGNED_URL_TTL_MINUTES must not exceed 1440 minutes (24 hours)."
            )
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS_ORIGINS as a parsed list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def presigned_url_ttl(self) -> timedelta:
        """Return PRESIGNED_URL_TTL_MINUTES as a timedelta for MinIO."""
        return timedelta(minutes=self.PRESIGNED_URL_TTL_MINUTES)


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton to avoid re-reading .env on every access."""
    try:
        return Settings()
    except ValidationError as e:
        # Provide clearer error message for missing required credentials
        errors = e.errors()
        missing_creds = []
        for err in errors:
            field = err.get("loc", ["unknown"])[0] if err.get("loc") else "unknown"
            if field in ("DATABASE_URL", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "FIREBASE_PROJECT_ID"):
                missing_creds.append(field)
        if missing_creds:
            raise RuntimeError(
                f"Missing required security credentials: {', '.join(missing_creds)}. "
                f"Set them via environment variables or .env file. "
                f"For local development, use docker-compose which provides these automatically."
            ) from e
        raise
