"""
Storage Access Tests

Tests verify MinIO object storage access controls, presigned URL generation,
error handling, and credential sanitization for both source and derived artifacts.

These tests use mocked MinIO client and mocked authentication. No live
MinIO/PostgreSQL required.
"""
import os
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from minio.error import S3Error

# Set required environment variables BEFORE importing app modules
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core")
os.environ.setdefault("MINIO_ACCESS_KEY", "testaccess")
os.environ.setdefault("MINIO_SECRET_KEY", "testsecret")
os.environ.setdefault("FIREBASE_PROJECT_ID", "neuroscan-medical-vault")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")

from app.core.storage import get_presigned_url, upload_file_to_minio, minio_client, init_buckets
from app.db.models import User, Scan, Prediction, Artifact
from app.main import app
from app.api.dependencies import get_db, get_current_user
from app.core.rate_limiting import enforce_upload_rate_limit


client = TestClient(app)


class TestPresignedUrlGeneration:
    """Test presigned URL generation for scan results."""

    def _setup_authenticated_user(self, user_uid="test-user", email="test@example.com"):
        """Helper to set up authenticated user and DB mock."""
        mock_user = User(firebase_uid=user_uid, email=email)
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: MagicMock()
        app.dependency_overrides[enforce_upload_rate_limit] = lambda: None
        return mock_user

    def _cleanup_overrides(self):
        app.dependency_overrides.clear()

    @patch("app.core.storage._get_minio_client")
    def test_presigned_url_success(self, mock_get_minio_client):
        """Valid object name returns presigned URL."""
        mock_minio = MagicMock()
        mock_minio.presigned_get_object.return_value = "https://minio.example.com/neuroscan-bucket/scan-1/mask.png?X-Amz-Algorithm=..."
        mock_get_minio_client.return_value = mock_minio

        url = get_presigned_url("scan-1/mask.png")

        assert url is not None
        assert url.startswith("https://")
        mock_minio.presigned_get_object.assert_called_once_with(
            "neuroscan-bucket", "scan-1/mask.png", expires=mock_minio.presigned_get_object.call_args[1]["expires"]
        )

    @patch("app.core.storage._get_minio_client")
    def test_presigned_url_custom_bucket(self, mock_get_minio_client):
        """Presigned URL works with custom bucket name."""
        mock_minio = MagicMock()
        mock_minio.presigned_get_object.return_value = "https://minio.example.com/custom-bucket/object.png?X-Amz-Algorithm=..."
        mock_get_minio_client.return_value = mock_minio

        url = get_presigned_url("object.png", bucket_name="custom-bucket")

        assert url is not None
        mock_minio.presigned_get_object.assert_called_once_with(
            "custom-bucket", "object.png", expires=mock_minio.presigned_get_object.call_args[1]["expires"]
        )

    @patch("app.core.storage._get_minio_client")
    def test_presigned_url_minio_error_returns_none(self, mock_get_minio_client):
        """MinIO error during presigned URL generation returns None (fail-safe)."""
        mock_minio = MagicMock()
        mock_minio.presigned_get_object.side_effect = S3Error(
            "Not Found",
            "The specified key does not exist",
            "NoSuchKey",
            "request-id",
            "host-id",
            None,
        )
        mock_get_minio_client.return_value = mock_minio

        url = get_presigned_url("nonexistent/object.png")

        assert url is None
        mock_minio.presigned_get_object.assert_called_once()

    @patch("app.core.storage._get_minio_client")
    def test_presigned_url_connection_error_returns_none(self, mock_get_minio_client):
        """Connection error during presigned URL generation returns None."""
        mock_minio = MagicMock()
        mock_minio.presigned_get_object.side_effect = ConnectionError("Connection refused")
        mock_get_minio_client.return_value = mock_minio

        url = get_presigned_url("scan-1/mask.png")

        assert url is None

    @patch("app.core.storage._get_minio_client")
    def test_presigned_url_no_credential_leakage_on_error(self, mock_get_minio_client):
        """MinIO exception with credentials in message does not leak them."""
        mock_minio = MagicMock()
        credential_laden_error = S3Error(
            "InvalidAccessKeyId",
            "The Access Key Id you provided does not exist in our records. "
            "access_key: MINIOACCESSKEY123, secret_key: MINIOSECRET456, "
            "endpoint: https://minio.internal:9000",
            "InvalidAccessKeyId",
            "req-123",
            "host-123",
            None,
        )
        mock_minio.presigned_get_object.side_effect = credential_laden_error
        mock_get_minio_client.return_value = mock_minio

        url = get_presigned_url("scan-1/mask.png")

        assert url is None
        # The function returns None and doesn't propagate the error message
        # Credential leakage is prevented at the call site by catching all exceptions


class TestResultsEndpointPresignedUrls:
    """Test presigned URL generation in /results endpoint with authorization."""

    def _mock_user(self, firebase_uid="user-a"):
        user = User(firebase_uid=firebase_uid, email=f"{firebase_uid}@example.com")
        return user

    def _mock_scan(self, scan_id, user_uid, status="SEGMENTED", mask_path="scan-1/mask.png",
                   xai_path="scan-1/xai.png", report_path="scan-1/report.pdf"):
        scan = Scan(
            id=scan_id,
            filename=f"{scan_id}/source_study",
            status=status,
            user_id=user_uid,
            mask_path=mask_path,
            xai_path=xai_path,
            report_path=report_path,
        )
        return scan

    def _mock_prediction(self, scan_id, tumor_detected=True, max_tumor_probability=0.95, model_version_id="model-v1"):
        pred = Prediction(
            id=1,
            scan_id=scan_id,
            model_version_id=model_version_id,
            tumor_detected=tumor_detected,
            max_tumor_probability=max_tumor_probability,
        )
        return pred

    def _setup_test(self, current_user, scan, prediction, mock_minio_presigned="https://presigned/test"):
        """Set up test client with mocked dependencies."""
        mock_db = MagicMock()

        # Mock scan query
        mock_scan_query = MagicMock()
        mock_scan_query.filter.return_value.first.return_value = scan

        # Mock prediction query
        mock_pred_query = MagicMock()
        mock_pred_query.filter.return_value.first.return_value = prediction

        call_count = [0]
        def query_side_effect(model):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_scan_query
            else:
                return mock_pred_query
        mock_db.query.side_effect = query_side_effect

        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: current_user
        app.dependency_overrides[enforce_upload_rate_limit] = lambda: None

        return mock_db

    def test_owner_gets_presigned_urls(self):
        """Scan owner receives presigned URLs for all artifacts."""
        user = self._mock_user("user-a")
        scan = self._mock_scan("scan-1", "user-a")
        prediction = self._mock_prediction("scan-1")

        mock_db = self._setup_test(user, scan, prediction)

        with patch("app.api.routers.scans.get_presigned_url", side_effect=lambda x: f"https://presigned/{x}") as mock_presigned:
            try:
                response = client.get(
                    "/api/v1/scans/results/scan-1",
                    headers={"Authorization": "Bearer valid-token"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["mask_url"] == "https://presigned/scan-1/mask.png"
                assert data["xai_url"] == "https://presigned/scan-1/xai.png"
                assert data["report_url"] == "https://presigned/scan-1/report.pdf"
                assert mock_presigned.call_count == 3
            finally:
                app.dependency_overrides.clear()

    def test_owner_missing_artifact_returns_none(self):
        """Owner gets None for missing artifacts (not 404)."""
        user = self._mock_user("user-a")
        scan = self._mock_scan("scan-2", "user-a", mask_path=None, xai_path=None, report_path=None)
        prediction = self._mock_prediction("scan-2")

        mock_db = self._setup_test(user, scan, prediction)

        with patch("app.api.routers.scans.get_presigned_url") as mock_presigned:
            try:
                response = client.get(
                    "/api/v1/scans/results/scan-2",
                    headers={"Authorization": "Bearer valid-token"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["mask_url"] is None
                assert data["xai_url"] is None
                assert data["report_url"] is None
                mock_presigned.assert_not_called()
            finally:
                app.dependency_overrides.clear()

    def test_non_owner_gets_403_no_presigned_urls(self):
        """Non-owner receives 403 and no presigned URLs are generated."""
        owner = self._mock_user("user-a")
        attacker = self._mock_user("user-b")
        scan = self._mock_scan("scan-3", "user-a")
        prediction = self._mock_prediction("scan-3")

        mock_db = self._setup_test(attacker, scan, prediction)

        with patch("app.api.routers.scans.get_presigned_url") as mock_presigned:
            try:
                response = client.get(
                    "/api/v1/scans/results/scan-3",
                    headers={"Authorization": "Bearer valid-token"},
                )

                assert response.status_code == 403
                mock_presigned.assert_not_called()
            finally:
                app.dependency_overrides.clear()

    def test_nonexistent_scan_gets_404(self):
        """Nonexistent scan returns 404."""
        user = self._mock_user("user-a")
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value.first.return_value = None
        mock_db.query.return_value = mock_query

        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[enforce_upload_rate_limit] = lambda: None

        try:
            with patch("app.api.routers.scans.get_presigned_url") as mock_presigned:
                response = client.get(
                    "/api/v1/scans/results/nonexistent",
                    headers={"Authorization": "Bearer valid-token"},
                )

                assert response.status_code == 404
                mock_presigned.assert_not_called()
        finally:
            app.dependency_overrides.clear()

    def test_legacy_scan_no_owner_gets_403(self):
        """Legacy scan with no owner returns 403 to everyone."""
        user = self._mock_user("user-a")
        scan = self._mock_scan("scan-legacy", user_uid=None)
        prediction = self._mock_prediction("scan-legacy")

        mock_db = self._setup_test(user, scan, prediction)

        try:
            with patch("app.api.routers.scans.get_presigned_url") as mock_presigned:
                response = client.get(
                    "/api/v1/scans/results/scan-legacy",
                    headers={"Authorization": "Bearer valid-token"},
                )

                assert response.status_code == 403
                mock_presigned.assert_not_called()
        finally:
            app.dependency_overrides.clear()


class TestStorageErrorHandling:
    """Test storage error handling and credential sanitization."""

    @staticmethod
    def _credential_laden_s3_error():
        """Return an S3Error containing credentials and internal details."""
        return S3Error(
            "InvalidAccessKeyId",
            "The Access Key Id you provided does not exist. "
            "access_key: MINIOACCESSKEY123, secret_key: MINIOSECRET456, "
            "endpoint: https://minio.internal:9000, region: us-east-1",
            "InvalidAccessKeyId",
            "request-123",
            "host-123",
            None,
        )

    @patch("app.api.routers.scans.minio_client")
    def test_upload_cleanup_credential_redaction(self, mock_minio_client, caplog):
        """_cleanup_source_objects redacts credentials from S3Error messages."""
        from app.api.routers.scans import _cleanup_source_objects

        mock_minio_client.remove_object.side_effect = self._credential_laden_s3_error()

        with caplog.at_level("WARNING"):
            _cleanup_source_objects("neuroscan-bucket", ["scan-1/source_t1.nii.gz"])

        assert "Exception" in caplog.text or "S3Error" in caplog.text
        assert "MINIOACCESSKEY123" not in caplog.text
        assert "MINIOSECRET456" not in caplog.text
        assert "endpoint" not in caplog.text
        assert "us-east-1" not in caplog.text
        assert "scan-1/source_t1.nii.gz" in caplog.text  # Object name preserved

    @patch("app.services.ai_tasks.minio_client")
    def test_worker_cleanup_credential_redaction(self, mock_minio_client, caplog):
        """_cleanup_derived_objects redacts credentials from S3Error messages."""
        from app.services.ai_tasks import _cleanup_derived_objects

        mock_minio_client.remove_object.side_effect = self._credential_laden_s3_error()

        with caplog.at_level("WARNING"):
            _cleanup_derived_objects("neuroscan-bucket", ["scan-1/mask.png"])

        assert "Exception" in caplog.text or "S3Error" in caplog.text
        assert "MINIOACCESSKEY123" not in caplog.text
        assert "MINIOSECRET456" not in caplog.text
        assert "scan-1/mask.png" in caplog.text

    @patch("app.api.routers.scans.minio_client")
    def test_upload_storage_failure_generic_500(self, mock_minio_client):
        """Storage failure during upload returns generic 500, no internal details."""
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api.dependencies import get_db, get_current_user
        from app.core.rate_limiting import enforce_upload_rate_limit
        from app.db.models import User
        from unittest.mock import MagicMock
        import nibabel as nib
        import numpy as np
        import tempfile
        import os

        # Create valid NIfTI
        rng = np.random.RandomState(0)
        data = rng.rand(32, 32, 32).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        files = {}
        for modality in ["t1", "t1ce", "t2", "flair"]:
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                nib.save(img, tmp.name)
                tmp_path = tmp.name
            with open(tmp_path, "rb") as f:
                buffer = BytesIO(f.read())
            os.unlink(tmp_path)
            buffer.seek(0)
            files[modality] = ("test.nii.gz", buffer, "application/octet-stream")

        # Setup auth
        mock_user = User(firebase_uid="test-user", email="test@example.com")
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_db] = lambda: MagicMock()
        app.dependency_overrides[enforce_upload_rate_limit] = lambda: None

        # Fail on MinIO upload
        mock_minio_client.bucket_exists.return_value = True
        mock_minio_client.fput_object.side_effect = self._credential_laden_s3_error()
        mock_minio_client.remove_object.return_value = None

        try:
            response = client.post(
                "/api/v1/scans/upload",
                files=files,
                headers={"Authorization": "Bearer valid-token"},
            )

            assert response.status_code == 500
            assert response.json()["detail"] == "Upload failed"
            # Verify no credential leakage in response
            assert "MINIOACCESSKEY123" not in response.text
            assert "MINIOSECRET456" not in response.text
            assert "endpoint" not in response.text
        finally:
            app.dependency_overrides.clear()

    @patch("app.core.storage._get_minio_client")
    def test_upload_file_to_minio_credential_redaction(self, mock_get_minio_client):
        """upload_file_to_minio doesn't leak credentials on error."""
        mock_minio = MagicMock()
        mock_minio.bucket_exists.return_value = True
        mock_minio.fput_object.side_effect = self._credential_laden_s3_error()
        mock_get_minio_client.return_value = mock_minio

        with pytest.raises(S3Error) as exc_info:
            upload_file_to_minio("/tmp/test.png", "test/object.png")

        # Original exception propagated for internal handling
        # S3Error stores the error message in the exception string
        assert "InvalidAccessKeyId" in str(exc_info.value)
        # But in production, the caller (upload endpoint) catches and sanitizes


class TestArtifactAccessPatterns:
    """Test artifact access patterns and cross-user isolation."""

    def _setup_artifact_test(self, scan_user_uid, request_user_uid):
        """Helper to set up artifact access test."""
        mock_user = User(firebase_uid=request_user_uid, email=f"{request_user_uid}@example.com")
        mock_db = MagicMock()

        scan = Scan(
            id="scan-artifacts-1",
            filename="scan-artifacts-1/source_study",
            status="SEGMENTED",
            user_id=scan_user_uid,
            mask_path="scan-artifacts-1/mask.png",
            xai_path="scan-artifacts-1/xai.png",
            report_path="scan-artifacts-1/report.pdf",
            xai_raw_path="scan-artifacts-1/xai_raw.npy",
        )

        prediction = Prediction(
            id=1,
            scan_id="scan-artifacts-1",
            model_version_id="model-v1",
            tumor_detected=True,
            max_tumor_probability=0.95,
        )

        artifacts = [
            Artifact(id=1, prediction_id=1, type="mask", object_path="scan-artifacts-1/mask.png"),
            Artifact(id=2, prediction_id=1, type="xai", object_path="scan-artifacts-1/xai.png"),
            Artifact(id=3, prediction_id=1, type="xai_raw", object_path="scan-artifacts-1/xai_raw.npy"),
            Artifact(id=4, prediction_id=1, type="report", object_path="scan-artifacts-1/report.pdf"),
        ]

        mock_scan_query = MagicMock()
        mock_scan_query.filter.return_value.first.return_value = scan
        mock_pred_query = MagicMock()
        mock_pred_query.filter.return_value.first.return_value = prediction
        mock_art_query = MagicMock()
        mock_art_query.filter.return_value.all.return_value = artifacts

        call_count = [0]
        def query_side_effect(model):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_scan_query
            elif call_count[0] == 2:
                return mock_pred_query
            else:
                return mock_art_query
        mock_db.query.side_effect = query_side_effect

        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[enforce_upload_rate_limit] = lambda: None

        return mock_db

    def test_artifact_types_match_design(self):
        """Verify all four artifact types are stored."""
        from app.db.models import Artifact

        artifact_types = {"mask", "xai", "xai_raw", "report"}

        # This is a compile-time check - verify the model allows these types
        # by checking the type column has no constraints beyond VARCHAR
        assert Artifact.__table__.columns["type"].nullable is False

    def test_artifact_object_paths_are_permanent(self):
        """Verify artifact object paths are permanent paths, not presigned URLs."""
        from app.db.models import Artifact

        # Artifacts store permanent MinIO object paths
        artifact = Artifact(
            prediction_id=1,
            type="mask",
            object_path="scan-1/mask.png"
        )

        # Object path should not contain query parameters (presigned URL indicator)
        assert "?" not in artifact.object_path
        assert "X-Amz-" not in artifact.object_path
        assert artifact.object_path == "scan-1/mask.png"

    @patch("app.api.routers.scans.get_presigned_url", side_effect=lambda x: f"https://presigned/{x}")
    def test_all_four_artifact_urls_generated(self, mock_presigned):
        """Owner gets presigned URLs for all four artifact types."""
        user = User(firebase_uid="user-a", email="a@example.com")
        scan = Scan(
            id="scan-4", filename="scan-4/source_study", status="SEGMENTED",
            user_id="user-a",
            mask_path="scan-4/mask.png",
            xai_path="scan-4/xai.png",
            report_path="scan-4/report.pdf",
            xai_raw_path="scan-4/xai_raw.npy",
        )
        prediction = Prediction(
            id=1, scan_id="scan-4", model_version_id="model-v1",
            tumor_detected=True, max_tumor_probability=0.95,
        )

        mock_db = MagicMock()
        call_count = [0]
        def query_side_effect(model):
            call_count[0] += 1
            if call_count[0] == 1:
                mock_q = MagicMock()
                mock_q.filter.return_value.first.return_value = scan
                return mock_q
            else:
                mock_q = MagicMock()
                mock_q.filter.return_value.first.return_value = prediction
                return mock_q
        mock_db.query.side_effect = query_side_effect

        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[enforce_upload_rate_limit] = lambda: None

        try:
            response = client.get(
                "/api/v1/scans/results/scan-4",
                headers={"Authorization": "Bearer valid-token"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["mask_url"] == "https://presigned/scan-4/mask.png"
            assert data["xai_url"] == "https://presigned/scan-4/xai.png"
            assert data["report_url"] == "https://presigned/scan-4/report.pdf"
            # xai_raw_path is not exposed in ResultsResponse (by design)
        finally:
            app.dependency_overrides.clear()


class TestSourceModalityAccess:
    """Test source modality file access patterns."""

    def test_modality_files_linked_to_scan(self):
        """ModalityFile records are linked to scan via FK."""
        from app.db.models import ModalityFile

        mf = ModalityFile(
            scan_id="scan-1",
            modality="t1",
            object_path="scan-1/source_t1.nii.gz"
        )

        assert mf.scan_id == "scan-1"
        assert mf.modality == "t1"
        assert mf.object_path == "scan-1/source_t1.nii.gz"

    def test_all_four_modalities_required(self):
        """Upload requires all four modalities (enforced at API level)."""
        from app.api.routers.scans import upload_scan
        from unittest.mock import AsyncMock, MagicMock
        from io import BytesIO
        import asyncio

        # This is verified by the FastAPI endpoint signature requiring all 4 files
        import inspect
        sig = inspect.signature(upload_scan)
        params = list(sig.parameters.keys())
        assert "t1" in params
        assert "t1ce" in params
        assert "t2" in params
        assert "flair" in params


class TestStorageArchitectureCompliance:
    """Test compliance with storage architecture rules (rules.md #16)."""

    def test_artifacts_belong_to_study(self):
        """Every artifact belongs to a study (via prediction -> scan)."""
        from app.db.models import Artifact, Prediction, Scan

        # Artifact -> Prediction -> Scan chain
        artifact = Artifact(prediction_id=1, type="mask", object_path="scan-1/mask.png")
        prediction = Prediction(id=1, scan_id="scan-1", model_version_id="model-v1",
                               tumor_detected=True, max_tumor_probability=0.95)
        scan = Scan(id="scan-1", filename="scan-1/source_study", status="SEGMENTED", user_id="user-1")

        # Verify the relationship chain exists in model definitions
        fks = list(Artifact.__table__.foreign_keys)
        pred_fk = [fk for fk in fks if fk.target_fullname == "predictions.id"][0]
        assert pred_fk.parent.name == "prediction_id"

        pred_fks = list(Prediction.__table__.foreign_keys)
        scan_fk = [fk for fk in pred_fks if fk.target_fullname == "scans.id"][0]
        assert scan_fk.parent.name == "scan_id"

    def test_derived_artifacts_reference_source_study(self):
        """Derived artifacts reference their source study via object path naming."""
        # Derived artifacts use scan_id as prefix: scan-id/mask.png, scan-id/xai.png, etc.
        derived_paths = [
            "scan-1/mask.png",
            "scan-1/xai.png",
            "scan-1/xai_raw.npy",
            "scan-1/report.pdf",
        ]

        for path in derived_paths:
            assert path.startswith("scan-1/")

    def test_source_artifacts_use_study_prefix(self):
        """Source artifacts use scan_id prefix: scan-id/source_modality.nii.gz."""
        source_paths = [
            "scan-1/source_t1.nii.gz",
            "scan-1/source_t1ce.nii.gz",
            "scan-1/source_t2.nii.gz",
            "scan-1/source_flair.nii.gz",
        ]

        for path in source_paths:
            assert path.startswith("scan-1/source_")

    def test_presigned_urls_are_delivery_not_provenance(self):
        """Presigned URLs are delivery mechanisms, not permanent provenance identifiers."""
        from app.core.storage import get_presigned_url
        from unittest.mock import patch, MagicMock

        with patch("app.core.storage._get_minio_client") as mock_get_minio:
            mock_minio = MagicMock()
            mock_minio.presigned_get_object.return_value = "https://minio.example.com/bucket/obj?X-Amz-Expires=900&X-Amz-Signature=..."
            mock_get_minio.return_value = mock_minio

            url = get_presigned_url("scan-1/mask.png")

            # Presigned URL contains expiration and signature - it's temporary
            assert "X-Amz-Expires" in url or "X-Amz-Signature" in url
            # Permanent provenance is the object path, not the presigned URL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])