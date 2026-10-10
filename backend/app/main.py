"""
NeuroScan AI — Application Entrypoint.
This module is strictly responsible for:
  1. FastAPI app initialization
  2. CORS middleware registration
  3. Router inclusion

Database schema is managed by Alembic migrations.
Startup does not create or drop tables.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routers.scans import router as scans_router

# ── Ensure all model definitions are imported so Base.metadata is populated ──
from app.db import models  # noqa: F401

settings = get_settings()

# ── FastAPI application ──
app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION)

# ── CORS middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Router registration ──
app.include_router(scans_router)