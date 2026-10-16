"""
Focused tests for SG3-F5-E: rate limiting and resource-exhaustion controls.

These tests are deterministic: they use an in-memory fake Redis client, mock
authentication, never contact real services, and never read production
credentials. Environment variables required by `Settings` are set before the
app is imported (mirroring `tests/test_auth.py`).

SG3-F6-C: The ``TestRedisIntegration`` class below adds mandatory
real-Redis integration tests that connect to a genuine Redis service
(provisioned by CI or local docker-compose) and are NOT mocked.
"""
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import redis
from fastapi import HTTPException
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

# Set required environment variables BEFORE importing app modules.
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core")
os.environ.setdefault("MINIO_ACCESS_KEY", "testaccess")
os.environ.setdefault("MINIO_SECRET_KEY", "testsecret")
os.environ.setdefault("FIREBASE_PROJECT_ID", "neuroscan-medical-vault")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")

from app.core.config import Settings, get_settings  # noqa: E402
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

    def test_redis_require_auth_defaults_to_false(self):
        """Development default: Redis auth enforcement is disabled, so an
        unauthenticated REDIS_URL is accepted."""
        s = self._settings()
        assert s.RATE_LIMIT_REDIS_REQUIRE_AUTH is False
        # Unauthenticated default URL is accepted when enforcement is off.
        assert s.REDIS_URL == "redis://localhost:6379/0"

    def test_redis_require_auth_true_requires_password_in_url(self):
        """When enforcement is enabled, REDIS_URL without credentials is rejected."""
        with pytest.raises(ValueError) as exc_info:
            self._settings(
                RATE_LIMIT_REDIS_REQUIRE_AUTH=True,
                REDIS_URL="redis://localhost:6379/0",
            )
        # Credentials must never appear in the error message.
        error_text = str(exc_info.value)
        assert "redis://localhost:6379/0" not in error_text

    def test_redis_require_auth_true_accepts_authenticated_url(self):
        """When enforcement is enabled, a REDIS_URL with credentials is accepted."""
        s = self._settings(
            RATE_LIMIT_REDIS_REQUIRE_AUTH=True,
            REDIS_URL="redis://default:test-secret@localhost:6379/0",
        )
        assert s.RATE_LIMIT_REDIS_REQUIRE_AUTH is True

    def test_redis_require_auth_true_accepts_rediss_scheme(self):
        """When enforcement is enabled, rediss:// URLs with credentials are accepted."""
        s = self._settings(
            RATE_LIMIT_REDIS_REQUIRE_AUTH=True,
            REDIS_URL="rediss://default:test-secret@localhost:6379/0",
        )
        assert s.RATE_LIMIT_REDIS_REQUIRE_AUTH is True

    def test_redis_require_auth_true_rejects_empty_credentials(self):
        """When enforcement is enabled, a URL with an empty credential component
        (e.g., redis://@host) is rejected."""
        with pytest.raises(ValueError):
            self._settings(
                RATE_LIMIT_REDIS_REQUIRE_AUTH=True,
                REDIS_URL="redis://@localhost:6379/0",
            )


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


class TestRedisIntegration:
    """Real-Redis integration tests for the upload rate limiter.

    These tests connect to a genuine Redis service (provisioned by CI or local
    docker-compose) and are NOT mocked. They are mandatory in CI: the GitHub
    Actions workflow provisions a ``redis:7-alpine`` service container and sets
    ``REDIS_URL`` plus ``RATE_LIMIT_REDIS_REQUIRE_AUTH=true``, so these tests
    run against real authenticated Redis on every push/PR.

    Tests cover both unauthenticated Redis (SG3-F6-C, when enforcement is
    disabled) and authenticated Redis (SG3-F6-D, when enforcement is enabled):

    * Correct credentials allow requests; exceeding the limit yields 429 with
      Retry-After.
    * Incorrect credentials fail closed with a generic 503 and no credential
      leakage.
    * Missing credentials are rejected when auth enforcement is enabled.
    * Unauthenticated requests still receive 401 before rate limiting.
    * Rate-limit cleanup deletes only namespaced keys, never FLUSHDB.

    Local developers without Redis can opt out by setting
    ``NEUROSCAN_SKIP_REDIS_INTEGRATION=1``; the tests are never silently
    skipped in CI because that environment variable is not set there.
    """

    # Authenticated Redis (used when RATE_LIMIT_REDIS_REQUIRE_AUTH=true in CI)
    REDIS_URL_AUTHENTICATED = "redis://:redis_ci_test_password_99@localhost:6379/0"
    REDIS_URL_WRONG_PASSWORD = "redis://:wrong_password@localhost:6379/0"
    # Unauthenticated Redis (development default when enforcement is disabled)
    REDIS_URL_AVAILABLE = "redis://localhost:6379/0"
    REDIS_URL_UNAVAILABLE = "redis://localhost:1/0"  # dead port, connection refused

    @pytest.fixture(autouse=True)
    def _reset_limiter_and_settings(self):
        """Clear the cached limiter and settings before/after each test."""
        reset_rate_limiter_for_tests()
        get_settings.cache_clear()
        yield
        reset_rate_limiter_for_tests()
        get_settings.cache_clear()

    @pytest.fixture
    def _skip_if_no_redis(self, request):
        """Opt-out for local developers without Redis; never skips in CI."""
        if os.environ.get("NEUROSCAN_SKIP_REDIS_INTEGRATION") == "1":
            pytest.skip("NEUROSCAN_SKIP_REDIS_INTEGRATION=1; no local Redis")

    def _make_limiter(self, redis_url: str) -> RedisRateLimiter:
        """Create a RedisRateLimiter connected to the given Redis URL."""
        settings = get_settings()
        redis_client = redis.Redis.from_url(
            redis_url,
            socket_connect_timeout=1,
            socket_timeout=1,
            health_check_interval=1,
        )
        return RedisRateLimiter(
            redis_client=redis_client,
            prefix=settings.RATE_LIMIT_KEY_PREFIX,
            max_requests=settings.RATE_LIMIT_UPLOAD_MAX_REQUESTS,
            window_seconds=settings.RATE_LIMIT_UPLOAD_WINDOW_SECONDS,
        )

    def _cleanup_rate_limit_keys(self, redis_url: str) -> None:
        """Delete only rate-limit namespace keys from a real Redis instance.

        Uses SCAN to find keys matching the rate-limit prefix and deletes them
        individually. Never uses FLUSHDB, so unrelated keys in the same database
        are preserved.
        """
        prefix = f"{get_settings().RATE_LIMIT_KEY_PREFIX}*"
        client = redis.Redis.from_url(redis_url, socket_connect_timeout=1)
        try:
            keys: list = []
            cursor = 0
            while True:
                cursor, batch = client.scan(cursor=cursor, match=prefix, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break
            if keys:
                client.delete(*keys)
        finally:
            client.close()

    def _upload_files(self):
        return {
            "t1": ("t1.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t1ce": ("t1ce.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t2": ("t2.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "flair": ("flair.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
        }

    # ------------------------------------------------------------------
    # 429 against real Redis
    # ------------------------------------------------------------------

    def test_429_when_limit_exceeded_real_redis(
        self, monkeypatch, _clean_dependency_overrides, _skip_if_no_redis
    ):
        """Exhausting the per-user limit against real Redis yields 429 +
        Retry-After. Uses authenticated Redis when enforcement is enabled."""
        settings = get_settings()
        mock_user = User(
            firebase_uid="redis-integration-user", email="redis@example.com"
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: MagicMock()

        # Use authenticated URL when enforcement is enabled, unauthenticated
        # otherwise (development default).
        if settings.RATE_LIMIT_REDIS_REQUIRE_AUTH:
            redis_url = self.REDIS_URL_AUTHENTICATED
        else:
            redis_url = self.REDIS_URL_AVAILABLE

        limiter = self._make_limiter(redis_url)
        monkeypatch.setattr(
            "app.core.rate_limiting.get_rate_limiter", lambda: limiter
        )

        try:
            # Exhaust the counter directly against real Redis.
            for _ in range(settings.RATE_LIMIT_UPLOAD_MAX_REQUESTS):
                allowed, _ = limiter.check(mock_user.firebase_uid)
                assert allowed is True

            # The next request must be rate limited.
            response = client.post(
                "/api/v1/scans/upload",
                files=self._upload_files(),
                headers={"Authorization": "Bearer any-token"},
            )
            assert response.status_code == 429
            body = response.json()
            assert body["detail"] == "Rate limit exceeded. Please try again later."
            assert "Retry-After" in response.headers
            retry_after = int(response.headers["Retry-After"])
            assert 1 <= retry_after <= settings.RATE_LIMIT_UPLOAD_WINDOW_SECONDS
        finally:
            self._cleanup_rate_limit_keys(redis_url)

    def test_429_when_limit_exceeded_authenticated_redis(
        self, monkeypatch, _clean_dependency_overrides, _skip_if_no_redis
    ):
        """Correct credentials allow requests; exceeding the limit yields
        429 + Retry-After against real authenticated Redis."""
        settings = get_settings()
        mock_user = User(
            firebase_uid="redis-auth-integration-user", email="redis-auth@example.com"
        )
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: MagicMock()

        limiter = self._make_limiter(self.REDIS_URL_AUTHENTICATED)
        monkeypatch.setattr(
            "app.core.rate_limiting.get_rate_limiter", lambda: limiter
        )

        try:
            # Exhaust the counter directly against real Redis.
            for _ in range(settings.RATE_LIMIT_UPLOAD_MAX_REQUESTS):
                allowed, _ = limiter.check(mock_user.firebase_uid)
                assert allowed is True

            # The next request must be rate limited.
            response = client.post(
                "/api/v1/scans/upload",
                files=self._upload_files(),
                headers={"Authorization": "Bearer any-token"},
            )
            assert response.status_code == 429
            body = response.json()
            assert body["detail"] == "Rate limit exceeded. Please try again later."
            assert "Retry-After" in response.headers
            retry_after = int(response.headers["Retry-After"])
            assert 1 <= retry_after <= settings.RATE_LIMIT_UPLOAD_WINDOW_SECONDS
        finally:
            self._cleanup_rate_limit_keys(self.REDIS_URL_AUTHENTICATED)

    # ------------------------------------------------------------------
    # 503 when Redis is unavailable
    # ------------------------------------------------------------------

    def test_503_when_redis_unavailable(
        self, monkeypatch, _clean_dependency_overrides, _skip_if_no_redis
    ):
        """Unavailable Redis backend yields a generic 503, no detail leaked."""
        mock_user = User(firebase_uid="redis-503-user", email="503@example.com")
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: MagicMock()

        limiter = self._make_limiter(self.REDIS_URL_UNAVAILABLE)
        monkeypatch.setattr(
            "app.core.rate_limiting.get_rate_limiter", lambda: limiter
        )

        response = client.post(
            "/api/v1/scans/upload",
            files=self._upload_files(),
            headers={"Authorization": "Bearer any-token"},
        )

        assert response.status_code == 503
        body = response.json()
        assert body["detail"] == "Service temporarily unavailable"
        # No Redis failure details, credentials, or internal text in the body.
        response_text = response.text.lower()
        assert "connection" not in response_text
        assert "refused" not in response_text
        assert "redis" not in response_text
        assert "localhost" not in response_text
        assert "6379" not in response_text

    def test_503_when_redis_wrong_password(
        self, monkeypatch, _clean_dependency_overrides, _skip_if_no_redis
    ):
        """Incorrect Redis credentials fail closed with a generic 503 and
        no credential leakage in the response body."""
        mock_user = User(firebase_uid="redis-wrong-pw-user", email="wrongpw@example.com")
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: MagicMock()

        limiter = self._make_limiter(self.REDIS_URL_WRONG_PASSWORD)
        monkeypatch.setattr(
            "app.core.rate_limiting.get_rate_limiter", lambda: limiter
        )

        response = client.post(
            "/api/v1/scans/upload",
            files=self._upload_files(),
            headers={"Authorization": "Bearer any-token"},
        )

        assert response.status_code == 503
        body = response.json()
        assert body["detail"] == "Service temporarily unavailable"
        # No Redis failure details, credentials, or internal text in the body.
        response_text = response.text.lower()
        assert "connection" not in response_text
        assert "refused" not in response_text
        assert "auth" not in response_text
        assert "password" not in response_text
        assert "redis" not in response_text
        assert "localhost" not in response_text
        assert "6379" not in response_text
        assert "redis_ci_test_password_99" not in response_text
        assert "wrong_password" not in response_text
        assert "connection refused" not in response_text.lower()
        assert "redis" not in response_text.lower()

    def test_auth_before_rate_limit_real_redis(
        self, _clean_dependency_overrides, _skip_if_no_redis
    ):
        """Unauthenticated requests are rejected with 401 before rate limiting."""
        # Do not override get_current_user; no auth header sent.
        response = client.post("/api/v1/scans/upload", files=self._upload_files())
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired authentication token"

    def test_auth_before_rate_limit_authenticated_redis(
        self, _clean_dependency_overrides, _skip_if_no_redis
    ):
        """Unauthenticated requests are rejected with 401 before rate
        limiting is checked, even against authenticated Redis."""
        # Do not override get_current_user; no auth header sent.
        response = client.post("/api/v1/scans/upload", files=self._upload_files())
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired authentication token"

    # ------------------------------------------------------------------
    # No credential leakage on 503
    # ------------------------------------------------------------------

    def test_no_credential_leakage_on_503(
        self, monkeypatch, _clean_dependency_overrides, _skip_if_no_redis
    ):
        """The 503 body contains no Redis connection details or credentials."""
        mock_user = User(firebase_uid="redis-leak-user", email="leak@example.com")
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: MagicMock()

        limiter = self._make_limiter(self.REDIS_URL_UNAVAILABLE)
        monkeypatch.setattr(
            "app.core.rate_limiting.get_rate_limiter", lambda: limiter
        )

        response = client.post(
            "/api/v1/scans/upload",
            files=self._upload_files(),
            headers={"Authorization": "Bearer any-token"},
        )

        assert response.status_code == 503
        body = response.json()
        assert body["detail"] == "Service temporarily unavailable"
        # Explicitly assert no sensitive strings appear anywhere in the response.
        assert "redis://" not in response.text
        assert "localhost:1" not in response.text
        assert "connection refused" not in response.text.lower()
        assert "redis" not in response.text.lower()

    def test_no_credential_leakage_on_wrong_password_503(
        self, monkeypatch, _clean_dependency_overrides, _skip_if_no_redis
    ):
        """The 503 body contains no Redis credentials or connection details."""
        mock_user = User(firebase_uid="redis-leak-user", email="leak@example.com")
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: MagicMock()

        limiter = self._make_limiter(self.REDIS_URL_WRONG_PASSWORD)
        monkeypatch.setattr(
            "app.core.rate_limiting.get_rate_limiter", lambda: limiter
        )

        response = client.post(
            "/api/v1/scans/upload",
            files=self._upload_files(),
            headers={"Authorization": "Bearer any-token"},
        )

        assert response.status_code == 503
        body = response.json()
        assert body["detail"] == "Service temporarily unavailable"
        # Explicitly assert no sensitive strings appear anywhere in the response.
        assert "redis://" not in response.text
        assert "localhost:1" not in response.text
        assert "redis_ci_test_password_99" not in response.text
        assert "wrong_password" not in response.text
        assert "connection refused" not in response.text.lower()
        assert "redis" not in response.text.lower()

    def test_cleanup_preserves_unrelated_keys(self, _skip_if_no_redis):
        """Rate-limit cleanup deletes only namespaced keys, leaving unrelated
        keys intact."""
        settings = get_settings()
        # Use authenticated URL when enforcement is enabled.
        if settings.RATE_LIMIT_REDIS_REQUIRE_AUTH:
            redis_url = self.REDIS_URL_AUTHENTICATED
        else:
            redis_url = self.REDIS_URL_AVAILABLE

        client_redis = redis.Redis.from_url(
            redis_url, socket_connect_timeout=1
        )
        try:
            # Insert an unrelated key outside the rate-limit namespace.
            unrelated_key = "unrelated:survival:test-key"
            client_redis.set(unrelated_key, "survive")

            # Run rate-limit cleanup.
            self._cleanup_rate_limit_keys(redis_url)

            # Unrelated key must survive.
            assert client_redis.get(unrelated_key) == b"survive"
        finally:
            # Best-effort cleanup of the test's own unrelated key.
            try:
                client_redis.delete("unrelated:survival:test-key")
            finally:
                client_redis.close()
