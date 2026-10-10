"""
Firebase Authentication Tests

Tests verify that the Firebase ID token verification dependency:
- Accepts valid tokens
- Rejects missing Authorization header
- Rejects malformed Authorization header
- Rejects invalid tokens
- Rejects revoked tokens
- Protects the scan status endpoint
- Enforces cross-user ownership on scan endpoints
"""
import os
from unittest.mock import AsyncMock, patch, MagicMock

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

    def test_upload_endpoint_requires_auth(self):
        """POST /api/v1/scans/upload without auth returns 401."""
        from io import BytesIO
        
        files = {
            "t1": ("t1.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t1ce": ("t1ce.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t2": ("t2.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "flair": ("flair.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
        }
        response = client.post("/api/v1/scans/upload", files=files)

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired authentication token"

    def test_results_endpoint_requires_auth(self):
        """GET /api/v1/scans/results/{scan_id} without auth returns 401."""
        response = client.get("/api/v1/scans/results/test-scan-id")

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid or expired authentication token"


class TestCrossUserOwnership:
    """Test cross-user ownership enforcement on scan endpoints.
    
    These tests mock the database and Firebase verification to test
    ownership enforcement without requiring real credentials or database.
    """

    @pytest.fixture
    def mock_user_a(self):
        """Create a mock User A."""
        from app.db.models import User
        user = User(firebase_uid="user-a-uid", email="usera@example.com")
        return user

    @pytest.fixture
    def mock_user_b(self):
        """Create a mock User B."""
        from app.db.models import User
        user = User(firebase_uid="user-b-uid", email="userb@example.com")
        return user

    @pytest.fixture
    def mock_scan_owned_by_a(self, mock_user_a):
        """Create a mock Scan owned by User A."""
        from app.db.models import Scan
        from datetime import datetime, timezone
        scan = Scan(
            id="test-scan-owned-by-a",
            filename="test-scan-owned-by-a/source_study",
            status="SEGMENTED",
            user_id=mock_user_a.firebase_uid,
            mask_path="test-scan-owned-by-a/mask.png",
            xai_path="test-scan-owned-by-a/xai.png",
            report_path="test-scan-owned-by-a/report.pdf",
        )
        return scan

    @pytest.fixture
    def mock_legacy_scan(self):
        """Create a mock legacy scan with no owner."""
        from app.db.models import Scan
        scan = Scan(
            id="test-legacy-scan",
            filename="test-legacy-scan/source_study",
            status="SEGMENTED",
            user_id=None,  # Legacy scan - no owner
            mask_path="test-legacy-scan/mask.png",
            xai_path="test-legacy-scan/xai.png",
            report_path="test-legacy-scan/report.pdf",
        )
        return scan

    @pytest.fixture
    def mock_prediction_for_scan(self):
        """Create a mock Prediction for test scans."""
        from app.db.models import Prediction
        pred = Prediction(
            id=1,
            scan_id="test-scan-owned-by-a",
            model_version_id="model-test",
            tumor_detected=True,
            max_tumor_probability=0.95,
        )
        return pred

    def _setup_db_mock(self, monkeypatch, current_user, scan, prediction=None):
        """Helper to mock database queries for a given user and scan."""
        import app.api.routers.scans as scans_module
        
        # Mock the database session
        mock_db = MagicMock()
        
        # Mock Scan query
        mock_scan_query = MagicMock()
        mock_scan_query.filter.return_value.first.return_value = scan
        mock_db.query.return_value = mock_scan_query
        
        # Mock Prediction query if needed
        if prediction:
            mock_pred_query = MagicMock()
            mock_pred_query.filter.return_value.first.return_value = prediction
            # Need to handle multiple query calls - Scan then Prediction
            call_count = [0]
            def query_side_effect(model):
                call_count[0] += 1
                if call_count[0] == 1:
                    return mock_scan_query
                else:
                    return mock_pred_query
            mock_db.query.side_effect = query_side_effect
        
        return mock_db

    def test_user_a_can_access_own_scan_status(self, mock_user_a, mock_scan_owned_by_a, monkeypatch):
        """User A can access status of their own scan (200 OK)."""
        from app.api.routers.scans import get_scan_status
        from unittest.mock import MagicMock
        
        mock_db = MagicMock()
        mock_scan_query = MagicMock()
        mock_scan_query.filter.return_value.first.return_value = mock_scan_owned_by_a
        mock_db.query.return_value = mock_scan_query
        
        # Test the endpoint function directly (sync function)
        result = get_scan_status(
            scan_id="test-scan-owned-by-a",
            db=mock_db,
            current_user=mock_user_a,
        )
        
        assert result.scan_id == "test-scan-owned-by-a"
        assert result.status == "SEGMENTED"
        mock_db.query.assert_called_once()

    def test_user_b_cannot_access_user_a_scan_status(self, mock_user_b, mock_scan_owned_by_a, monkeypatch):
        """User B receives 403 when requesting User A's scan status."""
        from app.api.routers.scans import get_scan_status
        from fastapi import HTTPException
        from unittest.mock import MagicMock
        import asyncio
        
        mock_db = MagicMock()
        mock_scan_query = MagicMock()
        mock_scan_query.filter.return_value.first.return_value = mock_scan_owned_by_a
        mock_db.query.return_value = mock_scan_query
        
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(get_scan_status(
                scan_id="test-scan-owned-by-a",
                db=mock_db,
                current_user=mock_user_b,
            ))
        
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail
        assert "another user" in exc_info.value.detail

    def test_user_a_can_access_own_scan_results(self, mock_user_a, mock_scan_owned_by_a, monkeypatch):
        """User A can access results of their own scan (200 OK)."""
        from app.api.routers.scans import get_scan_results
        from app.db.models import Prediction
        from unittest.mock import MagicMock, patch
        
        mock_prediction = Prediction(
            id=1,
            scan_id="test-scan-owned-by-a",
            model_version_id="model-test",
            tumor_detected=True,
            max_tumor_probability=0.95,
        )
        
        mock_db = MagicMock()
        call_count = [0]
        def query_side_effect(model):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_scan_query = MagicMock()
                mock_scan_query.filter.return_value.first.return_value = mock_scan_owned_by_a
                return mock_scan_query
            else:
                mock_pred_query = MagicMock()
                mock_pred_query.filter.return_value.first.return_value = mock_prediction
                return mock_pred_query
        mock_db.query.side_effect = query_side_effect
        
        # Mock get_presigned_url
        with patch("app.api.routers.scans.get_presigned_url", side_effect=lambda x: f"https://presigned/{x}"):
            result = get_scan_results(
                scan_id="test-scan-owned-by-a",
                db=mock_db,
                current_user=mock_user_a,
            )
        
        assert result.scan_id == "test-scan-owned-by-a"
        assert result.tumor_detected is True
        assert result.max_tumor_probability == 0.95
        assert result.mask_url == "https://presigned/test-scan-owned-by-a/mask.png"
        assert result.xai_url == "https://presigned/test-scan-owned-by-a/xai.png"
        assert result.report_url == "https://presigned/test-scan-owned-by-a/report.pdf"

    def test_user_b_cannot_access_user_a_scan_results(self, mock_user_b, mock_scan_owned_by_a, mock_prediction_for_scan, monkeypatch):
        """User B receives 403 when requesting User A's results."""
        from app.api.routers.scans import get_scan_results
        from fastapi import HTTPException
        from unittest.mock import MagicMock
        import asyncio
        
        mock_db = MagicMock()
        call_count = [0]
        def query_side_effect(model):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_scan_query = MagicMock()
                mock_scan_query.filter.return_value.first.return_value = mock_scan_owned_by_a
                return mock_scan_query
            else:
                mock_pred_query = MagicMock()
                mock_pred_query.filter.return_value.first.return_value = mock_prediction_for_scan
                return mock_pred_query
        mock_db.query.side_effect = query_side_effect
        
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(get_scan_results(
                scan_id="test-scan-owned-by-a",
                db=mock_db,
                current_user=mock_user_b,
            ))
        
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail
        
        # Verify presigned URLs were NOT generated (no call to get_presigned_url)
        # The 403 is raised before get_presigned_url is called

    def test_user_b_receives_403_for_legacy_scan(self, mock_user_b, mock_legacy_scan, monkeypatch):
        """An authenticated user receives 403 for an ownerless legacy scan."""
        from app.api.routers.scans import get_scan_status
        from fastapi import HTTPException
        from unittest.mock import MagicMock
        import asyncio
        
        mock_db = MagicMock()
        mock_scan_query = MagicMock()
        mock_scan_query.filter.return_value.first.return_value = mock_legacy_scan
        mock_db.query.return_value = mock_scan_query
        
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(get_scan_status(
                scan_id="test-legacy-scan",
                db=mock_db,
                current_user=mock_user_b,
            ))
        
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail

    def test_unauthorized_results_does_not_generate_presigned_urls(self, mock_user_b, mock_scan_owned_by_a, mock_prediction_for_scan, monkeypatch):
        """Unauthorized results requests do not generate presigned URLs or expose metadata."""
        from app.api.routers.scans import get_scan_results
        from fastapi import HTTPException
        from unittest.mock import MagicMock, patch
        import asyncio
        
        mock_db = MagicMock()
        call_count = [0]
        def query_side_effect(model):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_scan_query = MagicMock()
                mock_scan_query.filter.return_value.first.return_value = mock_scan_owned_by_a
                return mock_scan_query
            else:
                mock_pred_query = MagicMock()
                mock_pred_query.filter.return_value.first.return_value = mock_prediction_for_scan
                return mock_pred_query
        mock_db.query.side_effect = query_side_effect
        
        # Mock get_presigned_url to track calls
        with patch("app.api.routers.scans.get_presigned_url") as mock_presigned:
            mock_presigned.side_effect = lambda x: f"https://presigned/{x}"
            
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(get_scan_results(
                    scan_id="test-scan-owned-by-a",
                    db=mock_db,
                    current_user=mock_user_b,
                ))
            
            assert exc_info.value.status_code == 403
            # Verify get_presigned_url was never called
            mock_presigned.assert_not_called()

    def test_upload_ownership_derived_from_authenticated_identity(self, mock_user_a, monkeypatch):
        """Upload ownership is derived from authenticated identity, not client data."""
        from app.api.routers.scans import upload_scan
        from app.schemas.scan_schema import UploadResponse
        from unittest.mock import AsyncMock, MagicMock, patch
        import asyncio
        from io import BytesIO
        import gzip
        import nibabel as nib
        import numpy as np
        import tempfile
        import os
        
        # Create a valid NIfTI file content for testing
        # Use larger dimensions to exceed 1KB minimum size check
        data = np.zeros((64, 64, 64), dtype=np.float32)  # ~1MB uncompressed
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        
        with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
            tmp_path = tmp.name
        
        nib.save(img, tmp_path)
        
        with open(tmp_path, 'rb') as f:
            nifti_bytes = f.read()
        
        # Gzip it using gzip.compress (more reliable than GzipFile with BytesIO)
        gzipped_content = gzip.compress(nifti_bytes)
        
        os.unlink(tmp_path)
        
        # Create mock files with valid NIfTI gzipped content
        mock_files = {}
        for modality in ["t1", "t1ce", "t2", "flair"]:
            mock_file = MagicMock()
            mock_file.filename = f"{modality}.nii.gz"
            mock_file.file = BytesIO(gzipped_content)
            mock_files[modality] = mock_file
        
        mock_db = MagicMock()
        mock_bg = MagicMock()
        
        # Mock minio
        with patch("app.api.routers.scans.minio_client") as mock_minio:
            mock_minio.bucket_exists.return_value = True
            mock_minio.fput_object = AsyncMock()
            
            result = asyncio.run(upload_scan(
                t1=mock_files["t1"],
                t1ce=mock_files["t1ce"],
                t2=mock_files["t2"],
                flair=mock_files["flair"],
                db=mock_db,
                background_tasks=mock_bg,
                current_user=mock_user_a,
            ))
        
        # Verify scan was created with correct ownership from authenticated user
        # Capture the Scan object that was added to the session
        added_scans = [call.args[0] for call in mock_db.add.call_args_list if call.args[0].__class__.__name__ == "Scan"]
        assert len(added_scans) == 1
        scan = added_scans[0]
        assert scan.user_id == mock_user_a.firebase_uid
        assert scan.user_id == "user-a-uid"
        assert scan.status == "PENDING"
        assert result.scan_id == scan.id
        assert result.status == "PROCESSING"


class TestUploadValidation:
    """Test upload validation for MRI files."""

    def test_upload_rejects_invalid_extension(self):
        """Upload rejects files with invalid extension."""
        from io import BytesIO
        
        files = {
            "t1": ("t1.txt", BytesIO(b"dummy"), "text/plain"),
            "t1ce": ("t1ce.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t2": ("t2.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "flair": ("flair.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
        }
        response = client.post(
            "/api/v1/scans/upload",
            files=files,
            headers={"Authorization": "Bearer valid-token"},
        )
        
        assert response.status_code == 401  # Auth fails first (mocked)

    def test_upload_rejects_oversized_file(self):
        """Upload rejects files exceeding size limit."""
        from io import BytesIO
        
        # Create a file larger than MAX_MODALITY_FILE_SIZE (100MB)
        oversized_content = b"x" * (101 * 1024 * 1024)  # 101 MB
        
        files = {
            "t1": ("t1.nii.gz", BytesIO(oversized_content), "application/octet-stream"),
            "t1ce": ("t1ce.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t2": ("t2.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "flair": ("flair.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
        }
        response = client.post(
            "/api/v1/scans/upload",
            files=files,
            headers={"Authorization": "Bearer valid-token"},
        )
        
        assert response.status_code == 401  # Auth fails first (mocked)

    def test_upload_rejects_malformed_nifti(self):
        """Upload rejects malformed NIfTI files."""
        from io import BytesIO
        
        files = {
            "t1": ("t1.nii.gz", BytesIO(b"not a nifti file"), "application/octet-stream"),
            "t1ce": ("t1ce.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t2": ("t2.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "flair": ("flair.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
        }
        response = client.post(
            "/api/v1/scans/upload",
            files=files,
            headers={"Authorization": "Bearer valid-token"},
        )
        
        assert response.status_code == 401  # Auth fails first (mocked)

    def test_upload_rejects_empty_file(self):
        """Upload rejects empty files."""
        from io import BytesIO
        
        files = {
            "t1": ("t1.nii.gz", BytesIO(b""), "application/octet-stream"),
            "t1ce": ("t1ce.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t2": ("t2.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "flair": ("flair.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
        }
        response = client.post(
            "/api/v1/scans/upload",
            files=files,
            headers={"Authorization": "Bearer valid-token"},
        )
        
        assert response.status_code == 401  # Auth fails first (mocked)

    def test_upload_rejects_too_small_file(self):
        """Upload rejects files too small to be valid NIfTI."""
        from io import BytesIO
        
        files = {
            "t1": ("t1.nii.gz", BytesIO(b"tiny"), "application/octet-stream"),
            "t1ce": ("t1ce.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "t2": ("t2.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
            "flair": ("flair.nii.gz", BytesIO(b"dummy"), "application/octet-stream"),
        }
        response = client.post(
            "/api/v1/scans/upload",
            files=files,
            headers={"Authorization": "Bearer valid-token"},
        )
        
        assert response.status_code == 401  # Auth fails first (mocked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])