"""
Focused tests for SG3-F5-E: rate limiting and resource-exhaustion controls.

These tests are deterministic: they use an in-memory fake Redis client, mock
authentication, never contact real services, and never read production
credentials. Environment variables required by `Settings` are set before the
app is imported (mirroring `tests/test_auth.py`).
"""
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

# Set required environment variables BEFORE importing app modules.
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core")
os.environ.setdefault("MINIO_ACCESS_KEY", "testaccess")
os.environ.setdefault("MINIO_SECRET_KEY", "testsecret")
os.environ.setdefault("FIREBASE_PROJECT_ID", "neuroscan-medical-vault")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")

from app.core.config import Settings  # noqa: E402
from app.core.rate_limiting import (  # noqa: E402
    RedisRateLimiter,
    _DisabledRateLimiter,
    _RateLimiter,
    enforce_upload_rate_limit,
    get_rate_limiter,
    reset_rate_limiter_for_tests,
)
from app.db.models import User  # noqa: E402
from app.main import app  # noqa: E402
from app.api.dependencies import get_db, get_current_user  # noqa: E402

client = TestClient(app)


class FakeRedis:
    """In-memory fake implementing the Redis commands used by RedisRateLimiter."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.ttls: dict[str, int] = {}
        self.eval_calls: list[tuple] = []
        self.should_raise: Exception | None = None

    def eval(self, script: str, numkeys: int, *args: object) -> list:
        self.eval_calls.append((script, numkeys, args))
        if self.should_raise is not None:
            raise self.should_raise
        key = args[0]
        limit = int(args[1])
        window = int(args[2])
        current = self.counters.get(key, 0) + 1
        self.counters[key] = current
        if current == 1:
            self.ttls[key] = window
        if current > limit:
            return [0, self.ttls.get(key, window)]
        return [1, -1]


class FakeLimiter(_RateLimiter):
    """Programmable fake limiter for dependency-level tests."""

    def __init__(self, allowed: bool = True, retry_after: int = -1) -> None:
        self.allowed = allowed
        self.retry_after = retry_after
        self.calls: list[str] = []

    def check(self, identifier: str):  # type: ignore[override]
        self.calls.append(identifier)
        return self.allowed, self.retry_after


@pytest.fixture(autouse=True)
def _clean_limiter_cache():
    """Clear the cached rate limiter before each test."""
    reset_rate_limiter_for_tests()
    yield
    reset_rate_limiter_for_tests()


@pytest.fixture
def _clean_dependency_overrides():
    """Restore FastAPI dependency overrides after endpoint tests."""
    original = dict(app.dependency_overrides)
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(original)


class TestRateLimitConfig:
    """Configuration validation for rate-limit settings."""

    def _settings(self, **overrides) -> Settings:
        base = {
            "DATABASE_URL": "postgresql+psycopg2://x:x@localhost:5432/x",
            "MINIO_ACCESS_KEY": "access",
            "MINIO_SECRET_KEY": "secret",
            "FIREBASE_PROJECT_ID": "proj",
            "CORS_ORIGINS": "http://localhost:3000",
        }
        base.update(overrides)
        return Settings(**base)

    def test_defaults(self):
        s = self._settings()
        assert s.RATE_LIMIT_ENABLED is True
        assert s.RATE_LIMIT_STORAGE_BACKEND == "redis"
        assert s.RATE_LIMIT_KEY_PREFIX == "neuroscan:ratelimit:"
        assert s.RATE_LIMIT_UPLOAD_MAX_REQUESTS == 10
        assert s.RATE_LIMIT_UPLOAD_WINDOW_SECONDS == 60

    def test_upload_max_requests_must_be_positive(self):
        with pytest.raises(ValueError):
            self._settings(RATE_LIMIT_UPLOAD_MAX_REQUESTS=0)

    def test_upload_window_seconds_must_be_positive(self):
        with pytest.raises(ValueError):
            self._settings(RATE_LIMIT_UPLOAD_WINDOW_SECONDS=0)

    def test_backend_must_be_redis(self):
        with pytest.raises(ValueError):
            self._settings(RATE_LIMIT_STORAGE_BACKEND="inmemory")


class TestRedisRateLimiter:
    """Unit tests for the Redis-backed rate limiter with a fake client."""

    def _make(self, fake: FakeRedis, **kwargs) -> RedisRateLimiter:
        return RedisRateLimiter(
            redis_client=fake,
            prefix="test:ratelimit:",
            max_requests=3,
            window_seconds=60,
            **kwargs,
        )

    def test_within_limit_allowed(self):
        fake = FakeRedis()
        limiter = self._make(fake)
        allowed, retry_after = limiter.check("user-a")
        assert allowed is True
        assert retry_after == -1
        assert fake.counters["test:ratelimit:upload:user-a"] == 1

    def test_exceeded_returns_false_and_retry_after(self):
        fake = FakeRedis()
        limiter = self._make(fake)
        for _ in range(3):
            allowed, _ = limiter.check("user-a")
            assert allowed is True
        allowed, retry_after = limiter.check("user-a")  # 4th request
        assert allowed is False
        assert retry_after == 60  # TTL of the current window

    def test_backend_failure_raises_503_without_leak(self):
        fake = FakeRedis()
        fake.should_raise = RedisConnectionError("connection refused")
        limiter = self._make(fake)
        with pytest.raises(HTTPException) as exc_info:
            limiter.check("user-a")
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Service temporarily unavailable"
        # No internal exception text leaked to the client.
        assert "connection refused" not in exc_info.value.detail
        assert "RedisConnectionError" not in exc_info.value.detail

    def test_keys_are_per_user_and_namespaced(self):
        fake = FakeRedis()
        limiter = self._make(fake)
        limiter.check("user-a")
        limiter.check("user-b")
        keys = {call[2][0] for call in fake.eval_calls}
        assert "test:ratelimit:upload:user-a" in keys
        assert "test:ratelimit:upload:user-b" in keys
        assert len(keys) == 2


class TestEnforceUploadRateLimitDependency:
    """Direct tests of the `enforce_upload_rate_limit` FastAPI dependency."""

    @pytest.mark.asyncio
    async def test_within_limit_passes(self, monkeypatch):
        fake = FakeLimiter(allowed=True, retry_after=-1)
        monkeypatch.setattr("app.core.rate_limiting.get_rate_limiter", lambda: fake)
        current_user = User(firebase_uid="user-ok", email="ok@example.com")
        # Should not raise.
        await enforce_upload_rate_limit(current_user=current_user)
        assert fake.calls == ["user-ok"]

    @pytest.mark.asyncio
    async def test_exceeded_raises_429_with_retry_after(self, monkeypatch):
        fake = FakeLimiter(allowed=False, retry_after=30)
        monkeypatch.setattr("app.core.rate_limiting.get_rate_limiter", lambda: fake)
        current_user = User(firebase_uid="user-limited", email="limited@example.com")
        with pytest.raises(HTTPException) as exc_info:
            await enforce_upload_rate_limit(current_user=current_user)
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail == "Rate limit exceeded. Please try again later."
        assert exc_info.value.headers["Retry-After"] == "30"

    @pytest.mark.asyncio
    async def test_redis_unavailable_raises_503(self, monkeypatch):
        fake_redis = FakeRedis()
        fake_redis.should_raise = RedisConnectionError("down")
        limiter = RedisRateLimiter(
            redis_client=fake_redis,
            prefix="test:ratelimit:",
            max_requests=5,
            window_seconds=60,
        )
        monkeypatch.setattr("app.core.rate_limiting.get_rate_limiter", lambda: limiter)
        current_user = User(firebase_uid="user-x", email="x@example.com")
        with pytest.raises(HTTPException) as exc_info:
            await enforce_upload_rate_limit(current_user=current_user)
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Service temporarily unavailable"


class TestRateLimitEndpoint:
    """Endpoint-level tests: rate limit is enforced on POST /api/v1/scans/upload."""

    def _upload_files(self):
        return {
            "t1": ("t1.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t1ce": ("t1ce.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t2": ("t2.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "flair": ("flair.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
        }

    def test_exceeded_upload_returns_429_with_retry_after(
        self, monkeypatch, _clean_dependency_overrides
    ):
        """Over the limit: 429 with generic body and Retry-After header."""
        mock_user = User(firebase_uid="user-limited", email="limited@example.com")
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: MagicMock()

        fake = FakeLimiter(allowed=False, retry_after=30)
        monkeypatch.setattr("app.core.rate_limiting.get_rate_limiter", lambda: fake)

        response = client.post(
            "/api/v1/scans/upload", files=self._upload_files(),
            headers={"Authorization": "Bearer any-token"},
        )

        assert response.status_code == 429
        body = response.json()
        assert body["detail"] == "Rate limit exceeded. Please try again later."
        assert response.headers["Retry-After"] == "30"
        # The endpoint body never runs, so no upload validation/MinIO is touched.

    def test_unauthenticated_request_rejected_before_rate_limit(
        self, _clean_dependency_overrides
    ):
        """Auth still runs first: no credentials -> 401, rate limit never checked."""
        # Do not override get_current_user and send no auth header.
        response = client.post("/api/v1/scans/upload", files=self._upload_files())
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired authentication token"

    def test_limiter_is_isolated_by_authenticated_user(self, monkeypatch):
        """Different authenticated users are tracked independently (unit level)."""
        fake = FakeLimiter(allowed=True)
        monkeypatch.setattr("app.core.rate_limiting.get_rate_limiter", lambda: fake)
        import asyncio

        asyncio.run(enforce_upload_rate_limit(current_user=User(firebase_uid="user-a", email="a@example.com")))
        asyncio.run(enforce_upload_rate_limit(current_user=User(firebase_uid="user-b", email="b@example.com")))
        assert fake.calls == ["user-a", "user-b"]


class TestDisabledRateLimiter:
    """When rate limiting is disabled, no backend contact is made and all pass."""

    def test_disabled_allows_everyone(self):
        limiter = _DisabledRateLimiter()
        assert limiter.enabled is False
        allowed, retry_after = limiter.check("any-user")
        assert allowed is True
