"""
Artifact Lifecycle Transition Tests

Tests verify the atomic artifact state transition mechanism using
PostgreSQL row-level locking (SELECT FOR UPDATE NOWAIT).
"""
import os
from pathlib import Path

# Set required environment variables BEFORE importing app modules
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core")
os.environ.setdefault("MINIO_ACCESS_KEY", "testaccess")
os.environ.setdefault("MINIO_SECRET_KEY", "testsecret")

import pytest
import threading
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.database import SessionLocal
from app.db.models import Scan, Prediction, ModelVersion, Artifact
from app.services.ai_tasks import transition_artifact_state, ArtifactStateTransition


def _create_scan(db: Session, scan_id: str, status: str = "PENDING") -> Scan:
    """Helper to create a scan record with given status."""
    scan = Scan(id=scan_id, filename=f"{scan_id}/source_study", status=status)
    db.add(scan)
    db.commit()
    return scan


def _create_prediction(db: Session, scan_id: str) -> Prediction:
    """Helper to create a prediction record."""
    pred = Prediction(
        scan_id=scan_id,
        model_version_id=None,  # FK is nullable
        tumor_detected=False,
        max_tumor_probability=0.0,
    )
    db.add(pred)
    db.commit()
    return pred


def _create_artifact(db: Session, prediction_id: int, artifact_type: str = "mask", status: str = "PENDING") -> Artifact:
    """Helper to create an artifact record."""
    artifact = Artifact(
        prediction_id=prediction_id,
        type=artifact_type,
        object_path=f"test/{artifact_type}.png",
        status=status,
    )
    db.add(artifact)
    db.commit()
    return artifact


def _get_artifact_status(db: Session, artifact_id: int) -> str | None:
    """Helper to get current artifact status - bypasses identity map with raw SQL."""
    row = db.execute(
        text("SELECT status FROM artifacts WHERE id = :artifact_id"),
        {"artifact_id": artifact_id}
    ).fetchone()
    return row.status if row else None


class TestArtifactLifecycle:
    """Test the artifact lifecycle transition mechanism."""

    def test_pending_to_complete_succeeds(self, db_session: Session):
        """PENDING artifact can transition to COMPLETE."""
        scan = _create_scan(db_session, "test-artifact-001")
        pred = _create_prediction(db_session, scan.id)
        artifact = _create_artifact(db_session, pred.id, "mask", "PENDING")
        
        result = transition_artifact_state(db_session, artifact.id, "COMPLETE")
        
        assert result.success is True
        assert result.reason == "TRANSITIONED"
        assert result.artifact_id == artifact.id
        assert result.from_state == "PENDING"
        assert result.to_state == "COMPLETE"
        assert _get_artifact_status(db_session, artifact.id) == "COMPLETE"

    def test_pending_to_failed_succeeds(self, db_session: Session):
        """PENDING artifact can transition to FAILED."""
        scan = _create_scan(db_session, "test-artifact-002")
        pred = _create_prediction(db_session, scan.id)
        artifact = _create_artifact(db_session, pred.id, "xai", "PENDING")
        
        result = transition_artifact_state(db_session, artifact.id, "FAILED")
        
        assert result.success is True
        assert result.reason == "TRANSITIONED"
        assert result.artifact_id == artifact.id
        assert result.from_state == "PENDING"
        assert result.to_state == "FAILED"
        assert _get_artifact_status(db_session, artifact.id) == "FAILED"

    def test_complete_to_failed_rejected(self, db_session: Session):
        """COMPLETE artifact cannot transition to FAILED (terminal state)."""
        scan = _create_scan(db_session, "test-artifact-003")
        pred = _create_prediction(db_session, scan.id)
        artifact = _create_artifact(db_session, pred.id, "mask", "COMPLETE")
        
        result = transition_artifact_state(db_session, artifact.id, "FAILED")
        
        assert result.success is False
        assert result.reason == "INVALID_TRANSITION"
        assert result.artifact_id == artifact.id
        assert result.from_state == "COMPLETE"
        assert result.to_state == "FAILED"
        # Status should remain COMPLETE
        assert _get_artifact_status(db_session, artifact.id) == "COMPLETE"

    def test_complete_to_pending_rejected(self, db_session: Session):
        """COMPLETE artifact cannot transition back to PENDING."""
        scan = _create_scan(db_session, "test-artifact-004")
        pred = _create_prediction(db_session, scan.id)
        artifact = _create_artifact(db_session, pred.id, "xai", "COMPLETE")
        
        result = transition_artifact_state(db_session, artifact.id, "PENDING")
        
        assert result.success is False
        assert result.reason == "INVALID_TRANSITION"
        assert result.artifact_id == artifact.id
        assert result.from_state == "COMPLETE"
        assert result.to_state == "PENDING"
        # Status should remain COMPLETE
        assert _get_artifact_status(db_session, artifact.id) == "COMPLETE"

    def test_failed_to_complete_rejected(self, db_session: Session):
        """FAILED artifact cannot transition to COMPLETE (terminal state)."""
        scan = _create_scan(db_session, "test-artifact-005")
        pred = _create_prediction(db_session, scan.id)
        artifact = _create_artifact(db_session, pred.id, "mask", "FAILED")
        
        result = transition_artifact_state(db_session, artifact.id, "COMPLETE")
        
        assert result.success is False
        assert result.reason == "INVALID_TRANSITION"
        assert result.artifact_id == artifact.id
        assert result.from_state == "FAILED"
        assert result.to_state == "COMPLETE"
        # Status should remain FAILED
        assert _get_artifact_status(db_session, artifact.id) == "FAILED"

    def test_failed_to_pending_rejected(self, db_session: Session):
        """FAILED artifact cannot transition back to PENDING."""
        scan = _create_scan(db_session, "test-artifact-006")
        pred = _create_prediction(db_session, scan.id)
        artifact = _create_artifact(db_session, pred.id, "xai", "FAILED")
        
        result = transition_artifact_state(db_session, artifact.id, "PENDING")
        
        assert result.success is False
        assert result.reason == "INVALID_TRANSITION"
        assert result.artifact_id == artifact.id
        assert result.from_state == "FAILED"
        assert result.to_state == "PENDING"
        # Status should remain FAILED
        assert _get_artifact_status(db_session, artifact.id) == "FAILED"

    def test_missing_artifact_rejected(self, db_session: Session):
        """Non-existent artifact returns NOT_FOUND."""
        result = transition_artifact_state(db_session, 999999, "COMPLETE")
        
        assert result.success is False
        assert result.reason == "NOT_FOUND"
        assert result.artifact_id == 999999

    def test_invalid_target_state_rejected(self, db_session: Session):
        """Unknown target state returns INVALID_STATE."""
        scan = _create_scan(db_session, "test-artifact-007")
        pred = _create_prediction(db_session, scan.id)
        artifact = _create_artifact(db_session, pred.id, "mask", "PENDING")
        
        result = transition_artifact_state(db_session, artifact.id, "UNKNOWN_STATE")
        
        assert result.success is False
        assert result.reason == "INVALID_STATE"
        assert result.artifact_id == artifact.id
        assert result.to_state == "UNKNOWN_STATE"

    def test_concurrent_transition_only_one_succeeds(self, db_session: Session):
        """Two concurrent transitions: exactly one succeeds."""
        scan = _create_scan(db_session, "test-artifact-008")
        pred = _create_prediction(db_session, scan.id)
        artifact = _create_artifact(db_session, pred.id, "mask", "PENDING")
        
        results = []
        barrier = threading.Barrier(2)
        
        def attempt_transition():
            local_db = SessionLocal()
            try:
                barrier.wait()  # Synchronize start
                result = transition_artifact_state(local_db, artifact.id, "COMPLETE")
                results.append(result)
                if result.success:
                    local_db.commit()
                else:
                    local_db.rollback()
            finally:
                local_db.close()
        
        t1 = threading.Thread(target=attempt_transition)
        t2 = threading.Thread(target=attempt_transition)
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        # Exactly one should succeed
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        
        assert len(successes) == 1, f"Expected 1 success, got {len(successes)}"
        assert len(failures) == 1, f"Expected 1 failure, got {len(failures)}"
        assert failures[0].reason == "INVALID_TRANSITION"
        # When lock is not available (NOWAIT), from_state is None because we couldn't read the row
        # The important invariant is that the artifact ends up in COMPLETE state
        assert _get_artifact_status(db_session, artifact.id) == "COMPLETE"


# Pytest fixture for database session
@pytest.fixture
def db_session() -> Session:
    """Provide a clean database session for each test."""
    db = SessionLocal()
    try:
        # Clean up any existing test artifacts
        db.query(Artifact).filter(Artifact.object_path.like("test/%")).delete()
        db.query(Prediction).filter(Prediction.scan_id.like("test-artifact-%")).delete()
        db.query(Scan).filter(Scan.id.like("test-artifact-%")).delete()
        db.commit()
        yield db
    finally:
        # Cleanup after test
        db.query(Artifact).filter(Artifact.object_path.like("test/%")).delete()
        db.query(Prediction).filter(Prediction.scan_id.like("test-artifact-%")).delete()
        db.query(Scan).filter(Scan.id.like("test-artifact-%")).delete()
        db.commit()
        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])