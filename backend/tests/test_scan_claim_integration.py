"""
Scan Claim Integration Tests

Tests verify that claim_scan_for_processing is correctly integrated
into process_scan_task to prevent duplicate processing.
"""
import os
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

# Set required environment variables BEFORE importing app modules
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core")
os.environ.setdefault("MINIO_ACCESS_KEY", "testaccess")
os.environ.setdefault("MINIO_SECRET_KEY", "testsecret")

import pytest
import threading
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import Scan, ModalityFile, Prediction, ModelVersion, Artifact
from app.services.ai_tasks import claim_scan_for_processing, process_scan_task, ScanClaimResult


def _mock_fget_object_creates_file(bucket: str, object_name: str, local_path: str):
    """Side effect for minio_client.fget_object that creates a minimal valid NIfTI file."""
    import nibabel as nib
    import numpy as np

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # Create a minimal valid NIfTI file with BraTS-like dimensions (240x240x155)
    # The reference image code loads slice 50, so we need at least 51 slices
    data = np.zeros((240, 240, 155), dtype=np.float32)
    # Add some non-zero data so it's not completely empty
    data[100:140, 100:140, 50] = 1.0

    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, local_path)


def _create_scan(db: Session, scan_id: str, status: str = "PENDING") -> Scan:
    """Helper to create a scan record with given status."""
    scan = Scan(id=scan_id, filename=f"{scan_id}/source_study", status=status)
    db.add(scan)
    db.commit()
    return scan


def _create_modality_files(db: Session, scan_id: str) -> dict:
    """Helper to create modality file records."""
    modality_objects = {}
    for modality in ["t1", "t1ce", "t2", "flair"]:
        object_name = f"{scan_id}/source_{modality}.nii.gz"
        modality_record = ModalityFile(
            scan_id=scan_id,
            modality=modality,
            object_path=object_name,
        )
        db.add(modality_record)
        modality_objects[modality] = object_name
    db.commit()
    return modality_objects


def _get_scan_status(db: Session, scan_id: str) -> str | None:
    """Helper to get current scan status."""
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    return scan.status if scan else None


def _get_prediction_count(db: Session, scan_id: str) -> int:
    """Helper to count predictions for a scan."""
    return db.query(Prediction).filter(Prediction.scan_id == scan_id).count()


def _get_artifact_count(db: Session, scan_id: str) -> int:
    """Helper to count artifacts for a scan."""
    pred = db.query(Prediction).filter(Prediction.scan_id == scan_id).first()
    if not pred:
        return 0
    return db.query(Artifact).filter(Artifact.prediction_id == pred.id).count()


class TestScanClaimIntegration:
    """Test the scan claim integration with process_scan_task."""

    @patch("app.services.ai_tasks.minio_client")
    @patch("app.services.ai_tasks._get_global_model")
    @patch("app.services.ai_tasks.preprocess_brats_study")
    @patch("app.services.ai_tasks.generate_gradient_saliency")
    @patch("app.services.ai_tasks.generate_clinical_overlays")
    @patch("app.services.ai_tasks.generate_segmentation_report")
    @patch("app.services.ai_tasks.upload_file_to_minio")
    def test_successful_processing_claims_pending_scan(
        self,
        mock_upload_file_to_minio,
        mock_generate_segmentation_report,
        mock_generate_clinical_overlays,
        mock_generate_gradient_saliency,
        mock_preprocess_brats_study,
        mock_get_global_model,
        mock_minio_client,
        db_session: Session,
    ):
        """Successful processing claims a PENDING scan before work starts."""
        scan_id = "test-integration-claim-001"
        _create_scan(db_session, scan_id, "PENDING")
        modality_objects = _create_modality_files(db_session, scan_id)

        # Setup mocks
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 1, 224, 224)  # Model output tensor [B, 1, H, W]
        mock_get_global_model.return_value = mock_model

        import numpy as np
        from ai_pipeline.preprocessing import PreprocessingConfig

        # Mock preprocessing output
        mock_image_tensor = torch.randn(1, 4, 224, 224)
        mock_mask_tensor = torch.randn(1, 1, 224, 224)
        mock_metadata = {"slice_index": 50}
        mock_preprocess_brats_study.return_value = (mock_image_tensor, mock_mask_tensor, mock_metadata)

        # Mock XAI
        mock_xai_tensor = np.random.rand(1, 1, 224, 224).astype(np.float32)
        from app.services.xai import XAIProvenance
        mock_provenance = XAIProvenance(model_checkpoint="test.pth", model_version="test-v1")
        mock_generate_gradient_saliency.return_value = (mock_xai_tensor, mock_provenance)

        # Mock overlays
        from PIL import Image
        mock_seg_pil = Image.new("RGB", (224, 224))
        mock_xai_pil = Image.new("RGB", (224, 224))
        mock_generate_clinical_overlays.return_value = (mock_seg_pil, mock_xai_pil, True, np.random.rand(224, 224))

        # Mock MinIO
        mock_minio_client.fget_object.side_effect = _mock_fget_object_creates_file

        # Run the task
        process_scan_task(scan_id, modality_objects)

        # Verify claim happened (scan status changed to PROCESSING during processing)
        scan_status = _get_scan_status(db_session, scan_id)
        assert scan_status == "SEGMENTED", f"Expected SEGMENTED, got {scan_status}"

        # Verify prediction created
        pred_count = _get_prediction_count(db_session, scan_id)
        assert pred_count == 1, f"Expected 1 prediction, got {pred_count}"

        # Verify artifacts created (4: mask, xai, xai_raw, report)
        artifact_count = _get_artifact_count(db_session, scan_id)
        assert artifact_count == 4, f"Expected 4 artifacts, got {artifact_count}"

        # Verify claim was attempted (preprocessing called means claim succeeded)
        mock_preprocess_brats_study.assert_called_once()

    @patch("app.services.ai_tasks.minio_client")
    @patch("app.services.ai_tasks._get_global_model")
    @patch("app.services.ai_tasks.preprocess_brats_study")
    def test_already_processing_scan_exits_early(
        self,
        mock_preprocess_brats_study,
        mock_get_global_model,
        mock_minio_client,
        db_session: Session,
    ):
        """A scan already PROCESSING exits without invoking expensive processing."""
        scan_id = "test-integration-claim-002"
        _create_scan(db_session, scan_id, "PROCESSING")
        modality_objects = _create_modality_files(db_session, scan_id)

        # Setup mocks
        mock_model = MagicMock()
        mock_get_global_model.return_value = mock_model

        # Run the task - should exit early due to claim failure
        process_scan_task(scan_id, modality_objects)

        # Verify expensive processing was NOT invoked
        mock_preprocess_brats_study.assert_not_called()
        mock_get_global_model.assert_not_called()
        mock_minio_client.fget_object.assert_not_called()

        # Status should remain PROCESSING
        scan_status = _get_scan_status(db_session, scan_id)
        assert scan_status == "PROCESSING"

    @patch("app.services.ai_tasks.minio_client")
    @patch("app.services.ai_tasks._get_global_model")
    @patch("app.services.ai_tasks.preprocess_brats_study")
    @patch("app.services.ai_tasks.generate_gradient_saliency")
    @patch("app.services.ai_tasks.generate_clinical_overlays")
    @patch("app.services.ai_tasks.generate_segmentation_report")
    @patch("app.services.ai_tasks.upload_file_to_minio")
    def test_duplicate_dispatch_no_duplicate_predictions(
        self,
        mock_upload_file_to_minio,
        mock_generate_segmentation_report,
        mock_generate_clinical_overlays,
        mock_generate_gradient_saliency,
        mock_preprocess_brats_study,
        mock_get_global_model,
        mock_minio_client,
        db_session: Session,
    ):
        """Duplicate dispatch does not create duplicate Predictions or Artifacts."""
        scan_id = "test-integration-claim-003"
        _create_scan(db_session, scan_id, "PENDING")
        modality_objects = _create_modality_files(db_session, scan_id)

        # Setup mocks
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 1, 224, 224)  # Model output tensor [B, 1, H, W]
        mock_get_global_model.return_value = mock_model

        import numpy as np

        mock_image_tensor = torch.randn(1, 4, 224, 224)
        mock_mask_tensor = torch.randn(1, 1, 224, 224)
        mock_metadata = {"slice_index": 50}
        mock_preprocess_brats_study.return_value = (mock_image_tensor, mock_mask_tensor, mock_metadata)

        mock_xai_tensor = np.random.rand(1, 1, 224, 224).astype(np.float32)
        from app.services.xai import XAIProvenance
        mock_provenance = XAIProvenance(model_checkpoint="test.pth", model_version="test-v1")
        mock_generate_gradient_saliency.return_value = (mock_xai_tensor, mock_provenance)

        from PIL import Image
        mock_seg_pil = Image.new("RGB", (224, 224))
        mock_xai_pil = Image.new("RGB", (224, 224))
        mock_generate_clinical_overlays.return_value = (mock_seg_pil, mock_xai_pil, True, np.random.rand(224, 224))

        mock_minio_client.fget_object.side_effect = _mock_fget_object_creates_file

        # Simulate concurrent dispatch with two threads
        results = []
        barrier = threading.Barrier(2)
        thread_exceptions = []

        def run_task():
            local_db = SessionLocal()
            try:
                barrier.wait()
                process_scan_task(scan_id, modality_objects)
                results.append("completed")
            except Exception as e:
                thread_exceptions.append(e)
                results.append(f"error: {e}")
            finally:
                local_db.close()

        t1 = threading.Thread(target=run_task)
        t2 = threading.Thread(target=run_task)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # No thread should have crashed with unhandled exceptions
        assert len(thread_exceptions) == 0, f"Thread exceptions occurred: {thread_exceptions}"

        # Verify exactly one prediction and one set of artifacts
        pred_count = _get_prediction_count(db_session, scan_id)
        assert pred_count == 1, f"Expected 1 prediction, got {pred_count}"

        artifact_count = _get_artifact_count(db_session, scan_id)
        assert artifact_count == 4, f"Expected 4 artifacts, got {artifact_count}"

        # Status should be SEGMENTED (completed by one worker)
        scan_status = _get_scan_status(db_session, scan_id)
        assert scan_status == "SEGMENTED"

        # Preprocessing should have been called only once (by the worker that won the claim)
        assert mock_preprocess_brats_study.call_count == 1

    @patch("app.services.ai_tasks.minio_client")
    @patch("app.services.ai_tasks._get_global_model")
    @patch("app.services.ai_tasks.preprocess_brats_study")
    def test_processing_failure_persists_failed(
        self,
        mock_preprocess_brats_study,
        mock_get_global_model,
        mock_minio_client,
        db_session: Session,
    ):
        """A processing failure persists FAILED using the existing error path."""
        scan_id = "test-integration-claim-004"
        _create_scan(db_session, scan_id, "PENDING")
        modality_objects = _create_modality_files(db_session, scan_id)

        # Setup mocks - make preprocessing fail
        mock_model = MagicMock()
        mock_get_global_model.return_value = mock_model
        mock_minio_client.fget_object = MagicMock()
        mock_preprocess_brats_study.side_effect = Exception("Preprocessing failed")

        # Run the task
        process_scan_task(scan_id, modality_objects)

        # Verify scan status is FAILED
        scan_status = _get_scan_status(db_session, scan_id)
        assert scan_status == "FAILED", f"Expected FAILED, got {scan_status}"

        # Verify no prediction created
        pred_count = _get_prediction_count(db_session, scan_id)
        assert pred_count == 0, f"Expected 0 predictions, got {pred_count}"

    @patch("app.services.ai_tasks.minio_client")
    @patch("app.services.ai_tasks._get_global_model")
    @patch("app.services.ai_tasks.preprocess_brats_study")
    @patch("app.services.ai_tasks.generate_gradient_saliency")
    @patch("app.services.ai_tasks.generate_clinical_overlays")
    @patch("app.services.ai_tasks.generate_segmentation_report")
    @patch("app.services.ai_tasks.upload_file_to_minio")
    def test_failed_scan_can_be_claimed_again(
        self,
        mock_upload_file_to_minio,
        mock_generate_segmentation_report,
        mock_generate_clinical_overlays,
        mock_generate_gradient_saliency,
        mock_preprocess_brats_study,
        mock_get_global_model,
        mock_minio_client,
        db_session: Session,
    ):
        """A FAILED scan can be claimed again (retry)."""
        scan_id = "test-integration-claim-005"
        _create_scan(db_session, scan_id, "FAILED")
        modality_objects = _create_modality_files(db_session, scan_id)

        # Setup mocks for successful processing
        import torch
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 1, 224, 224)  # Model output tensor [B, 1, H, W]
        mock_get_global_model.return_value = mock_model

        import torch
        import numpy as np

        mock_image_tensor = torch.randn(1, 4, 224, 224)
        mock_mask_tensor = torch.randn(1, 1, 224, 224)
        mock_metadata = {"slice_index": 50}
        mock_preprocess_brats_study.return_value = (mock_image_tensor, mock_mask_tensor, mock_metadata)

        mock_xai_tensor = np.random.rand(1, 1, 224, 224).astype(np.float32)
        from app.services.xai import XAIProvenance
        mock_provenance = XAIProvenance(model_checkpoint="test.pth", model_version="test-v1")
        mock_generate_gradient_saliency.return_value = (mock_xai_tensor, mock_provenance)

        from PIL import Image
        mock_seg_pil = Image.new("RGB", (224, 224))
        mock_xai_pil = Image.new("RGB", (224, 224))
        mock_generate_clinical_overlays.return_value = (mock_seg_pil, mock_xai_pil, True, np.random.rand(224, 224))

        mock_minio_client.fget_object.side_effect = _mock_fget_object_creates_file

        # Run the task directly - claim happens internally (FAILED -> PROCESSING)
        process_scan_task(scan_id, modality_objects)

        # Verify scan completed successfully
        scan_status = _get_scan_status(db_session, scan_id)
        assert scan_status == "SEGMENTED", f"Expected SEGMENTED, got {scan_status}"

        pred_count = _get_prediction_count(db_session, scan_id)
        assert pred_count == 1, f"Expected 1 prediction, got {pred_count}"

    @patch("app.services.ai_tasks.minio_client")
    @patch("app.services.ai_tasks._get_global_model")
    @patch("app.services.ai_tasks.preprocess_brats_study")
    @patch("app.services.ai_tasks.generate_gradient_saliency")
    @patch("app.services.ai_tasks.generate_clinical_overlays")
    @patch("app.services.ai_tasks.generate_segmentation_report")
    @patch("app.services.ai_tasks.upload_file_to_minio")
    def test_commit_failure_after_upload_rolls_back_artifacts(
        self,
        mock_upload_file_to_minio,
        mock_generate_segmentation_report,
        mock_generate_clinical_overlays,
        mock_generate_gradient_saliency,
        mock_preprocess_brats_study,
        mock_get_global_model,
        mock_minio_client,
        db_session: Session,
    ):
        """If DB commit fails after MinIO uploads, no Prediction or Artifact rows persist; Scan is marked FAILED."""
        scan_id = "test-integration-claim-006"
        _create_scan(db_session, scan_id, "PENDING")
        modality_objects = _create_modality_files(db_session, scan_id)

        # Setup mocks for successful processing up to commit
        import torch
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 1, 224, 224)
        mock_get_global_model.return_value = mock_model

        import torch
        import numpy as np

        mock_image_tensor = torch.randn(1, 4, 224, 224)
        mock_mask_tensor = torch.randn(1, 1, 224, 224)
        mock_metadata = {"slice_index": 50}
        mock_preprocess_brats_study.return_value = (mock_image_tensor, mock_mask_tensor, mock_metadata)

        mock_xai_tensor = np.random.rand(1, 1, 224, 224).astype(np.float32)
        from app.services.xai import XAIProvenance
        mock_provenance = XAIProvenance(model_checkpoint="test.pth", model_version="test-v1")
        mock_generate_gradient_saliency.return_value = (mock_xai_tensor, mock_provenance)

        from PIL import Image
        mock_seg_pil = Image.new("RGB", (224, 224))
        mock_xai_pil = Image.new("RGB", (224, 224))
        mock_generate_clinical_overlays.return_value = (mock_seg_pil, mock_xai_pil, True, np.random.rand(224, 224))

        mock_minio_client.fget_object.side_effect = _mock_fget_object_creates_file
        mock_upload_file_to_minio.return_value = None  # Uploads succeed

        # Inject commit failure on the primary session only.
        # The production code calls SessionLocal() twice:
        # 1. At the start (primary session)
        # 2. In the exception handler (fresh session for FAILED marking)
        # We make the first call return a session that fails on commit,
        # and the second call return a normal session.
        from app.db.database import SessionLocal
        call_count = [0]

        def failing_session_local():
            call_count[0] += 1
            session = SessionLocal()
            if call_count[0] == 1:
                # First call: primary session - make commit fail
                original_commit = session.commit
                def fail_commit():
                    raise Exception("Simulated commit failure")
                session.commit = fail_commit
            return session

        with patch("app.services.ai_tasks.SessionLocal", side_effect=failing_session_local):
            process_scan_task(scan_id, modality_objects)

        # Verify the failed transaction leaves no persisted Prediction or Artifact rows
        pred_count = _get_prediction_count(db_session, scan_id)
        assert pred_count == 0, f"Expected 0 predictions after rollback, got {pred_count}"

        artifact_count = _get_artifact_count(db_session, scan_id)
        assert artifact_count == 0, f"Expected 0 artifacts after rollback, got {artifact_count}"

        # Verify Scan is marked FAILED using the existing failure-handling pattern
        scan_status = _get_scan_status(db_session, scan_id)
        assert scan_status == "FAILED", f"Expected FAILED, got {scan_status}"

        # Verify no COMPLETE artifacts leaked (redundant but explicit)
        pred = db_session.query(Prediction).filter(Prediction.scan_id == scan_id).first()
        if pred:
            complete_artifacts = db_session.query(Artifact).filter(
                Artifact.prediction_id == pred.id,
                Artifact.status == "COMPLETE"
            ).count()
            assert complete_artifacts == 0, f"Expected 0 COMPLETE artifacts, got {complete_artifacts}"


# Pytest fixture for database session
@pytest.fixture
def db_session() -> Session:
    """Provide a clean database session for each test."""
    db = SessionLocal()
    try:
        # Clean up any existing test scans
        db.query(Artifact).filter(Artifact.object_path.like("test-%")).delete()
        db.query(Prediction).filter(Prediction.scan_id.like("test-%")).delete()
        db.query(ModalityFile).filter(ModalityFile.scan_id.like("test-%")).delete()
        db.query(Scan).filter(Scan.id.like("test-%")).delete()
        db.commit()
        yield db
    finally:
        # Cleanup after test
        db.query(Artifact).filter(Artifact.object_path.like("test-%")).delete()
        db.query(Prediction).filter(Prediction.scan_id.like("test-%")).delete()
        db.query(ModalityFile).filter(ModalityFile.scan_id.like("test-%")).delete()
        db.query(Scan).filter(Scan.id.like("test-%")).delete()
        db.commit()
        db.close()


# =============================================================================
# UPLOAD COMPENSATION TESTS
# =============================================================================

class TestUploadCompensation:
    """Test source upload failure compensation in the /upload endpoint."""

    def _create_upload_files(self):
        """Create mock UploadFile objects for testing."""
        from io import BytesIO
        from fastapi import UploadFile
        import tempfile
        import os

        # Create minimal valid NIfTI content
        import nibabel as nib
        import numpy as np
        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        files = {}
        for modality in ["t1", "t1ce", "t2", "flair"]:
            # Save to temp file first (nibabel doesn't support BytesIO directly on Windows)
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                nib.save(img, tmp.name)
                tmp_path = tmp.name

            # Read into BytesIO
            with open(tmp_path, "rb") as f:
                buffer = BytesIO(f.read())

            # Clean up temp file
            os.unlink(tmp_path)

            buffer.seek(0)
            files[modality] = UploadFile(
                filename=f"test_{modality}.nii.gz",
                file=buffer,
            )
        return files

    def _get_test_client_with_db(self, db_session):
        """Create a TestClient with the database dependency overridden."""
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api.dependencies import get_db

        def override_get_db():
            try:
                yield db_session
            finally:
                pass  # Don't close - fixture handles it

        app.dependency_overrides[get_db] = override_get_db
        client = TestClient(app)
        return client, app

    @patch("app.api.routers.scans.minio_client")
    @patch("app.services.ai_tasks.minio_client")
    @patch("fastapi.BackgroundTasks.add_task")
    @patch("uuid.uuid4")
    def test_successful_upload_retains_all_objects_and_persists_records(
        self,
        mock_uuid,
        mock_add_task,
        mock_ai_minio_client,
        mock_minio_client,
        db_session: Session,
    ):
        """Successful upload retains all four objects and persists Scan and ModalityFile records."""
        client, app = self._get_test_client_with_db(db_session)
        try:
            # Setup mocks
            mock_uuid.return_value = "test-upload-success-001"
            mock_minio_client.bucket_exists.return_value = True
            mock_minio_client.fput_object.return_value = None
            mock_ai_minio_client.bucket_exists.return_value = True
            mock_ai_minio_client.fput_object.return_value = None
            mock_ai_minio_client.fget_object = MagicMock()
            mock_add_task.return_value = None  # Don't actually run background task

            upload_files = self._create_upload_files()

            # Make request
            response = client.post(
                "/api/v1/scans/upload",
                files={
                    "t1": ("t1.nii.gz", upload_files["t1"].file, "application/octet-stream"),
                    "t1ce": ("t1ce.nii.gz", upload_files["t1ce"].file, "application/octet-stream"),
                    "t2": ("t2.nii.gz", upload_files["t2"].file, "application/octet-stream"),
                    "flair": ("flair.nii.gz", upload_files["flair"].file, "application/octet-stream"),
                },
            )

            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")

            assert response.status_code == 200
            data = response.json()
            scan_id = data["scan_id"]

            # Verify response
            assert data["status"] == "PROCESSING"
            assert "successfully" in data["message"].lower()

            # Verify MinIO uploads called for all 4 modalities
            assert mock_minio_client.fput_object.call_count == 4

            # Verify Scan record persisted
            scan = db_session.query(Scan).filter(Scan.id == scan_id).first()
            assert scan is not None
            assert scan.status == "PENDING"

            # Verify ModalityFile records persisted (4)
            modality_count = db_session.query(ModalityFile).filter(ModalityFile.scan_id == scan_id).count()
            assert modality_count == 4
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.routers.scans.minio_client")
    @patch("uuid.uuid4")
    def test_third_modality_failure_cleans_up_first_two(
        self,
        mock_uuid,
        mock_minio_client,
        db_session: Session,
    ):
        """Failure on third modality upload removes first two successfully uploaded objects."""
        client, app = self._get_test_client_with_db(db_session)
        try:
            # Setup mocks - fail on 3rd call (index 2 = t2)
            mock_uuid.return_value = "test-upload-fail-002"
            call_count = [0]
            def failing_fput_object(bucket, object_name, file_path):
                call_count[0] += 1
                if call_count[0] == 3:  # Third modality (t2) fails
                    raise Exception("MinIO upload failed")

            mock_minio_client.bucket_exists.return_value = True
            mock_minio_client.fput_object.side_effect = failing_fput_object
            mock_minio_client.remove_object.return_value = None

            upload_files = self._create_upload_files()

            # Make request
            response = client.post(
                "/api/v1/scans/upload",
                files={
                    "t1": ("t1.nii.gz", upload_files["t1"].file, "application/octet-stream"),
                    "t1ce": ("t1ce.nii.gz", upload_files["t1ce"].file, "application/octet-stream"),
                    "t2": ("t2.nii.gz", upload_files["t2"].file, "application/octet-stream"),
                    "flair": ("flair.nii.gz", upload_files["flair"].file, "application/octet-stream"),
                },
            )

            # Verify HTTP 500 returned
            assert response.status_code == 500

            # Verify first 2 uploads succeeded, 3rd failed
            assert mock_minio_client.fput_object.call_count == 3

            # Verify cleanup called for first 2 objects
            assert mock_minio_client.remove_object.call_count == 2

            # Verify no Scan record persisted (DB not reached or rolled back)
            scans = db_session.query(Scan).filter(Scan.id == "test-upload-fail-002").all()
            assert len(scans) == 0
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.routers.scans.minio_client")
    @patch("uuid.uuid4")
    def test_db_commit_failure_cleans_up_all_four_uploads(
        self,
        mock_uuid,
        mock_minio_client,
        db_session: Session,
    ):
        """Database commit failure rolls back DB records and attempts cleanup of all four uploaded objects."""
        client, app = self._get_test_client_with_db(db_session)
        try:
            # Setup mocks - all uploads succeed, DB commit fails
            mock_uuid.return_value = "test-upload-db-fail-003"
            mock_minio_client.bucket_exists.return_value = True
            mock_minio_client.fput_object.return_value = None
            mock_minio_client.remove_object.return_value = None

            # Create a session that fails on commit
            original_commit = db_session.commit
            def failing_commit():
                raise Exception("Simulated DB commit failure")
            db_session.commit = failing_commit

            upload_files = self._create_upload_files()

            # Make request
            response = client.post(
                "/api/v1/scans/upload",
                files={
                    "t1": ("t1.nii.gz", upload_files["t1"].file, "application/octet-stream"),
                    "t1ce": ("t1ce.nii.gz", upload_files["t1ce"].file, "application/octet-stream"),
                    "t2": ("t2.nii.gz", upload_files["t2"].file, "application/octet-stream"),
                    "flair": ("flair.nii.gz", upload_files["flair"].file, "application/octet-stream"),
                },
            )

            # Verify HTTP 500 returned
            assert response.status_code == 500

            # Verify all 4 uploads succeeded
            assert mock_minio_client.fput_object.call_count == 4

            # Verify cleanup attempted for all 4 objects
            assert mock_minio_client.remove_object.call_count == 4

            # Verify no Scan record persisted (rolled back)
            scans = db_session.query(Scan).filter(Scan.id == "test-upload-db-fail-003").all()
            assert len(scans) == 0

            # Verify no ModalityFile records persisted
            modality_files = db_session.query(ModalityFile).filter(ModalityFile.scan_id == "test-upload-db-fail-003").all()
            assert len(modality_files) == 0
        finally:
            # Restore original commit method so fixture cleanup works
            db_session.commit = original_commit
            app.dependency_overrides.clear()

    @patch("app.api.routers.scans.minio_client")
    @patch("uuid.uuid4")
    def test_cleanup_failure_does_not_mask_original_error(
        self,
        mock_uuid,
        mock_minio_client,
        db_session: Session,
    ):
        """Cleanup failure does not mask the original upload/DB error."""
        client, app = self._get_test_client_with_db(db_session)
        try:
            # Setup mocks - fail on 2nd upload, and cleanup also fails
            mock_uuid.return_value = "test-upload-cleanup-fail-004"
            call_count = [0]
            def failing_fput_object(bucket, object_name, file_path):
                call_count[0] += 1
                if call_count[0] == 2:  # Second modality (t1ce) fails
                    raise Exception("Original upload error")

            mock_minio_client.bucket_exists.return_value = True
            mock_minio_client.fput_object.side_effect = failing_fput_object
            mock_minio_client.remove_object.side_effect = Exception("Cleanup failed")

            upload_files = self._create_upload_files()

            # Make request
            response = client.post(
                "/api/v1/scans/upload",
                files={
                    "t1": ("t1.nii.gz", upload_files["t1"].file, "application/octet-stream"),
                    "t1ce": ("t1ce.nii.gz", upload_files["t1ce"].file, "application/octet-stream"),
                    "t2": ("t2.nii.gz", upload_files["t2"].file, "application/octet-stream"),
                    "flair": ("flair.nii.gz", upload_files["flair"].file, "application/octet-stream"),
                },
            )

            # Verify HTTP 500 returned
            assert response.status_code == 500

            # Verify original error message is in response (not cleanup error)
            data = response.json()
            assert "Original upload error" in str(data.get("detail", ""))
        finally:
            app.dependency_overrides.clear()

    @patch("app.api.routers.scans.minio_client")
    @patch("uuid.uuid4")
    def test_cleanup_ignores_already_missing_objects(
        self,
        mock_uuid,
        mock_minio_client,
        db_session: Session,
    ):
        """Cleanup treats already-missing objects as success (NoSuchKey)."""
        client, app = self._get_test_client_with_db(db_session)
        try:
            # Setup mocks - fail on 2nd upload, cleanup raises NoSuchKey for first object
            mock_uuid.return_value = "test-upload-nosuchkey-005"
            call_count = [0]
            def failing_fput_object(bucket, object_name, file_path):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise Exception("Upload failed")

            class S3Error(Exception):
                def __init__(self, message, code=None):
                    super().__init__(message)
                    self.code = code

            def remove_object_with_nosuchkey(bucket, object_name):
                raise S3Error("Not found", code="NoSuchKey")

            mock_minio_client.bucket_exists.return_value = True
            mock_minio_client.fput_object.side_effect = failing_fput_object
            mock_minio_client.remove_object.side_effect = remove_object_with_nosuchkey

            upload_files = self._create_upload_files()

            # Make request
            response = client.post(
                "/api/v1/scans/upload",
                files={
                    "t1": ("t1.nii.gz", upload_files["t1"].file, "application/octet-stream"),
                    "t1ce": ("t1ce.nii.gz", upload_files["t1ce"].file, "application/octet-stream"),
                    "t2": ("t2.nii.gz", upload_files["t2"].file, "application/octet-stream"),
                    "flair": ("flair.nii.gz", upload_files["flair"].file, "application/octet-stream"),
                },
            )

            # Verify HTTP 500 returned
            assert response.status_code == 500

            # Verify cleanup was attempted (NoSuchKey handled gracefully)
            assert mock_minio_client.remove_object.call_count == 1

            # No Scan record persisted
            scans = db_session.query(Scan).filter(Scan.id == "test-upload-nosuchkey-005").all()
            assert len(scans) == 0
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])