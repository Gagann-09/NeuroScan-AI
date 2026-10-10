"""
Firebase Authentication Tests

Tests verify that the Firebase ID token verification dependency:
- Accepts valid tokens
- Rejects missing Authorization header
- Rejects malformed Authorization header
- Rejects invalid tokens
- Rejects revoked tokens
- Protects the scan status endpoint
"""
import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Set required environment variables BEFORE importing app modules
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core")
os.environ.setdefault("MINIO_ACCESS_KEY", "testaccess")
os.environ.setdefault("MINIO_SECRET_KEY", "testsecret")
os.environ.setdefault("FIREBASE_PROJECT_ID", "neuroscan-medical-vault")

from app.core.security import verify_id_token
from app.main import app


client = TestClient(app)


class TestVerifyIdToken:
    """Test the verify_id_token dependency function."""

    @pytest.mark.asyncio
    async def test_verify_id_token_valid(self):
        """Valid token returns decoded claims."""
        mock_decoded = {
            "uid": "test-uid-123",
            "email": "test@example.com",
            "email_verified": True,
            "auth_time": 1699999999,
            "exp": 1700003599,
        }

        with patch("app.core.security.auth.verify_id_token", return_value=mock_decoded) as mock_verify:
            result = await verify_id_token(authorization="Bearer valid-token-123")

            assert result == mock_decoded
            mock_verify.assert_called_once_with("valid-token-123", check_revoked=True)

    @pytest.mark.asyncio
    async def test_verify_id_token_missing_header(self):
        """Missing Authorization header returns 401."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_id_token(authorization=None)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or expired authentication token"

    @pytest.mark.asyncio
    async def test_verify_id_token_empty_header(self):
        """Empty Authorization header returns 401."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_id_token(authorization="")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or expired authentication token"

    @pytest.mark.asyncio
    async def test_verify_id_token_malformed_header_no_bearer(self):
        """Authorization header without Bearer prefix returns 401."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_id_token(authorization="Token abc123")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or expired authentication token"

    @pytest.mark.asyncio
    async def test_verify_id_token_malformed_header_bearer_only(self):
        """Authorization header with only 'Bearer' returns 401."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_id_token(authorization="Bearer ")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or expired authentication token"

    @pytest.mark.asyncio
    async def test_verify_id_token_invalid(self):
        """Invalid token returns 401 with generic message."""
        from firebase_admin.auth import InvalidIdTokenError

        with patch("app.core.security.auth.verify_id_token", side_effect=InvalidIdTokenError("Invalid token")):
            with pytest.raises(HTTPException) as exc_info:
                await verify_id_token(authorization="Bearer invalid-token")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or expired authentication token"

    @pytest.mark.asyncio
    async def test_verify_id_token_expired(self):
        """Expired token returns 401 with generic message."""
        from firebase_admin.auth import ExpiredIdTokenError

        with patch("app.core.security.auth.verify_id_token", side_effect=ExpiredIdTokenError("Token expired", cause=None)):
            with pytest.raises(HTTPException) as exc_info:
                await verify_id_token(authorization="Bearer expired-token")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or expired authentication token"

    @pytest.mark.asyncio
    async def test_verify_id_token_revoked(self):
        """Revoked token returns 401 with generic message."""
        from firebase_admin.auth import RevokedIdTokenError

        with patch("app.core.security.auth.verify_id_token", side_effect=RevokedIdTokenError("Token revoked")):
            with pytest.raises(HTTPException) as exc_info:
                await verify_id_token(authorization="Bearer revoked-token")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or expired authentication token"

    @pytest.mark.asyncio
    async def test_verify_id_token_certificate_fetch_error(self):
        """Certificate fetch error returns 401 with generic message."""
        from firebase_admin.auth import CertificateFetchError

        with patch("app.core.security.auth.verify_id_token", side_effect=CertificateFetchError("Cert fetch failed", cause=None)):
            with pytest.raises(HTTPException) as exc_info:
                await verify_id_token(authorization="Bearer token")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or expired authentication token"

    @pytest.mark.asyncio
    async def test_verify_id_token_generic_firebase_error(self):
        """Any FirebaseError returns 401 with generic message."""
        from firebase_admin.exceptions import FirebaseError

        with patch("app.core.security.auth.verify_id_token", side_effect=FirebaseError(code="internal", message="Generic error")):
            with pytest.raises(HTTPException) as exc_info:
                await verify_id_token(authorization="Bearer token")

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid or expired authentication token"


class TestProtectedEndpoint:
    """Test that the scan status endpoint requires authentication."""

    def test_status_endpoint_requires_auth(self):
        """GET /api/v1/scans/status/{scan_id} without auth returns 401."""
        response = client.get("/api/v1/scans/status/test-scan-id")

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired authentication token"

    def test_status_endpoint_malformed_auth(self):
        """GET /api/v1/scans/status/{scan_id} with malformed auth returns 401."""
        response = client.get(
            "/api/v1/scans/status/test-scan-id",
            headers={"Authorization": "Token invalid"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired authentication token"

    def test_status_endpoint_invalid_token(self):
        """GET /api/v1/scans/status/{scan_id} with invalid token returns 401."""
        from firebase_admin.auth import InvalidIdTokenError

        with patch("app.core.security.auth.verify_id_token", side_effect=InvalidIdTokenError("Invalid")):
            response = client.get(
                "/api/v1/scans/status/test-scan-id",
                headers={"Authorization": "Bearer invalid-token"},
            )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired authentication token"

    def test_status_endpoint_revoked_token(self):
        """GET /api/v1/scans/status/{scan_id} with revoked token returns 401."""
        from firebase_admin.auth import RevokedIdTokenError

        with patch("app.core.security.auth.verify_id_token", side_effect=RevokedIdTokenError("Revoked")):
            response = client.get(
                "/api/v1/scans/status/test-scan-id",
                headers={"Authorization": "Bearer revoked-token"},
            )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired authentication token"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])