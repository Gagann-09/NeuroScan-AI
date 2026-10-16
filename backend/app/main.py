"""
NeuroScan AI — Application Entrypoint.
This module is strictly responsible for:
  1. FastAPI app initialization (including the managed lifespan)
  2. CORS middleware registration
  3. Router inclusion

Database schema is managed by Alembic migrations.
Startup does not create or drop tables.
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routers.scans import router as scans_router
from app.services.ai_tasks import _recovery_loop

# ── Ensure all model definitions are imported so Base.metadata is populated ──
from app.db import models  # noqa: F401

settings = get_settings()

# ── Managed lifecycle for the stale-PENDING recovery loop ──
_recovery_stop_event: asyncio.Event | None = None
_recovery_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the periodic recovery loop on startup and stop it cleanly on shutdown."""
    global _recovery_stop_event, _recovery_task
    _recovery_stop_event = asyncio.Event()
    _recovery_task = asyncio.create_task(_recovery_loop(_recovery_stop_event))
    try:
        yield
    finally:
        _recovery_stop_event.set()
        if _recovery_task is not None:
            await _recovery_task


# ── FastAPI application ──
app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION, lifespan=lifespan)

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
