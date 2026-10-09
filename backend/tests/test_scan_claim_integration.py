"""
Scan Claim Integration Tests

Tests verify that claim_scan_for_processing is correctly integrated
into process_scan_task to prevent duplicate processing.
"""
import os
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
    @patch("app.services.ai_tasks._get_or_create_model_version")
    def test_successful_processing_claims_pending_scan(
        self,
        mock_get_or_create_model_version,
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
        mock_model.return_value = MagicMock()
        mock_get_global_model.return_value = mock_model
        
        import torch
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
        
        # Mock model version
        mock_model_version = MagicMock(spec=ModelVersion)
        mock_model_version.id = "model-v1"
        mock_get_or_create_model_version.return_value = mock_model_version
        
        # Mock MinIO
        mock_minio_client.fget_object = MagicMock()
        
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
    @patch("app.services.ai_tasks._get_or_create_model_version")
    def test_duplicate_dispatch_no_duplicate_predictions(
        self,
        mock_get_or_create_model_version,
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
        mock_model.return_value = MagicMock()
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
        
        mock_model_version = MagicMock(spec=ModelVersion)
        mock_model_version.id = "model-v1"
        mock_get_or_create_model_version.return_value = mock_model_version
        
        mock_minio_client.fget_object = MagicMock()
        
        # Simulate concurrent dispatch with two threads
        results = []
        barrier = threading.Barrier(2)
        
        def run_task():
            local_db = SessionLocal()
            try:
                barrier.wait()
                process_scan_task(scan_id, modality_objects)
                results.append("completed")
            except Exception as e:
                results.append(f"error: {e}")
            finally:
                local_db.close()
        
        t1 = threading.Thread(target=run_task)
        t2 = threading.Thread(target=run_task)
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
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
    @patch("app.services.ai_tasks._get_or_create_model_version")
    def test_failed_scan_can_be_claimed_again(
        self,
        mock_get_or_create_model_version,
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
        mock_model = MagicMock()
        mock_model.return_value = MagicMock()
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
        
        mock_model_version = MagicMock(spec=ModelVersion)
        mock_model_version.id = "model-v1"
        mock_get_or_create_model_version.return_value = mock_model_version
        
        mock_minio_client.fget_object = MagicMock()
        
        # First, verify claim works on FAILED scan
        local_db = SessionLocal()
        try:
            result = claim_scan_for_processing(local_db, scan_id)
            assert result.success is True
            assert result.reason == "CLAIMED"
            local_db.commit()
        finally:
            local_db.close()
        
        # Now run the full task - should succeed
        process_scan_task(scan_id, modality_objects)
        
        # Verify scan completed successfully
        scan_status = _get_scan_status(db_session, scan_id)
        assert scan_status == "SEGMENTED", f"Expected SEGMENTED, got {scan_status}"
        
        pred_count = _get_prediction_count(db_session, scan_id)
        assert pred_count == 1, f"Expected 1 prediction, got {pred_count}"


# Pytest fixture for database session
@pytest.fixture
def db_session() -> Session:
    """Provide a clean database session for each test."""
    db = SessionLocal()
    try:
        # Clean up any existing test scans
        db.query(Artifact).filter(Artifact.object_path.like("test-integration-%")).delete()
        db.query(Prediction).filter(Prediction.scan_id.like("test-integration-%")).delete()
        db.query(ModalityFile).filter(ModalityFile.scan_id.like("test-integration-%")).delete()
        db.query(Scan).filter(Scan.id.like("test-integration-%")).delete()
        db.commit()
        yield db
    finally:
        # Cleanup after test
        db.query(Artifact).filter(Artifact.object_path.like("test-integration-%")).delete()
        db.query(Prediction).filter(Prediction.scan_id.like("test-integration-%")).delete()
        db.query(ModalityFile).filter(ModalityFile.scan_id.like("test-integration-%")).delete()
        db.query(Scan).filter(Scan.id.like("test-integration-%")).delete()
        db.commit()
        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])