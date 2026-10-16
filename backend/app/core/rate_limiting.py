"""
Rate limiting and resource-exhaustion controls for expensive API operations.

Provides a shared, production-suitable Redis-backed rate limiter so that
per-user counters are consistent across uvicorn workers and service
replicas. An in-memory/process-local limiter would NOT be shared across
workers and is therefore not offered as production protection.

Design decisions (see memory.md D-044):
- Backend: Redis, already part of the deployment topology
  (docker-compose `redis:7-alpine`, `REDIS_URL` setting, `redis` package).
  No new mandatory infrastructure dependency is introduced.
- Scope: `POST /api/v1/scans/upload` only -- the expensive endpoint that
  triggers scan processing (inference is dispatched as a background task).
  Status/results polling are cheap database reads and are deliberately not
  rate limited so normal scan monitoring is not broken.
- Keying: per authenticated user (Firebase `uid`). All scan endpoints require
  authentication (`get_current_user`), so the user is known before the
  limit is checked. IP-based limiting is intentionally not used: proxy
  handling is not established as trustworthy, and client-supplied forwarding
  headers must never be trusted.
- Limiter backend unavailable: expensive operations fail closed with a
  generic `503 Service Unavailable`. No internal details, credentials, or
  stack traces are leaked to clients.
- Response on exceeded limit: `429 Too Many Requests` with a `Retry-After`
  header (seconds until the current window resets) and a generic body.

All limits are configurable through `Settings` (app/core/config.py) and
validated at startup.
"""

import logging
from typing import Any, Tuple

from fastapi import Depends, HTTPException, status
from redis.exceptions import RedisError

from app.core.config import get_settings
from app.core.security import get_current_user
from app.db.models import User

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Interface for rate-limit backends."""

    enabled = True

    def check(self, identifier: str) -> Tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)`` for ``identifier``."""
        raise NotImplementedError


class _DisabledRateLimiter(_RateLimiter):
    """Rate limiting disabled: every request is allowed, no backend contact."""

    enabled = False

    def check(self, identifier: str) -> Tuple[bool, int]:  # noqa: ARG002
        return True, -1


# Lua script: atomically increment the counter for `key`, set the expiry on
# first use, and reject once the counter exceeds `limit`.
# Returns [allowed (0/1), retry_after (TTL in seconds when rejected)].
_RATE_LIMIT_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, window)
end
if current > limit then
    local ttl = redis.call('TTL', key)
    return {0, ttl}
end
return {1, -1}
"""


class RedisRateLimiter(_RateLimiter):
    """Redis-backed fixed-window rate limiter.

    Counters are shared across all workers and replicas that talk to the same
    Redis instance, which is required for correct limiting in a multi-worker
    or multi-replica deployment.
    """

    def __init__(
        self,
        redis_client: Any,
        prefix: str,
        max_requests: int,
        window_seconds: int,
    ) -> None:
        self._redis = redis_client
        self._prefix = prefix
        self._max_requests = max_requests
        self._window_seconds = window_seconds

    def check(self, identifier: str) -> Tuple[bool, int]:
        """Check whether ``identifier`` may make another request.

        Returns ``(allowed, retry_after_seconds)``. When not allowed,
        ``retry_after_seconds`` is the TTL of the current window (seconds
        until the counter resets).
        """
        key = f"{self._prefix}upload:{identifier}"
        try:
            result = self._redis.eval(
                _RATE_LIMIT_LUA,
                1,
                key,
                self._max_requests,
                self._window_seconds,
            )
        except RedisError as exc:
            # Fail closed: the limiter backend is unavailable, so refuse the
            # expensive operation rather than risk an unlimited burst.
            logger.error(
                "Rate-limit backend (Redis) unavailable; failing closed", exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service temporarily unavailable",
            ) from exc

        allowed = bool(int(result[0]))
        raw_retry_after = int(result[1])
        if allowed:
            # Not rate limited: no retry applies.
            retry_after = -1
        else:
            # Rate limited: retry after the current window resets (TTL).
            retry_after = raw_retry_after if raw_retry_after > 0 else self._window_seconds
        return allowed, retry_after


# Module-level cache of the configured limiter. `get_settings()` is already
# cached, so this is instantiated once per process.
_limiter: _RateLimiter | None = None


def get_rate_limiter() -> _RateLimiter:
    """Return the configured rate-limiter instance for the current process.

    Returns a :class:`RedisRateLimiter` backed by a lazy `redis.Redis` client
    (no connection is opened until the first command is issued), or a
    :class:`_DisabledRateLimiter` when rate limiting is disabled via settings.
    """
    global _limiter
    if _limiter is not None:
        return _limiter

    settings = get_settings()
    if not settings.RATE_LIMIT_ENABLED:
        _limiter = _DisabledRateLimiter()
        return _limiter

    import redis

    redis_client = redis.Redis.from_url(settings.REDIS_URL)
    _limiter = RedisRateLimiter(
        redis_client=redis_client,
        prefix=settings.RATE_LIMIT_KEY_PREFIX,
        max_requests=settings.RATE_LIMIT_UPLOAD_MAX_REQUESTS,
        window_seconds=settings.RATE_LIMIT_UPLOAD_WINDOW_SECONDS,
    )
    return _limiter


def reset_rate_limiter_for_tests() -> None:
    """Test-only helper to clear the cached limiter instance."""
    global _limiter
    _limiter = None


async def enforce_upload_rate_limit(
    current_user: User = Depends(get_current_user),
) -> None:
    """FastAPI dependency: enforce the per-user scan-upload rate limit.

    Runs after authentication, so the limit is keyed on the authenticated
    Firebase UID. Unauthenticated requests are rejected (401) by
    `get_current_user` before the limit is ever checked. Raises `429` when
    the limit is exceeded and `503` when the Redis backend is unavailable.
    """
    limiter = get_rate_limiter()
    allowed, retry_after = limiter.check(current_user.firebase_uid)
    if not allowed:
        logger.warning(
            "Rate limit exceeded for user %s", current_user.firebase_uid
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )
