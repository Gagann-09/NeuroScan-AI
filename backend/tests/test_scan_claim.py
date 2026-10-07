"""
Scan Claim/Locking Mechanism Tests

Tests verify the atomic claim function for scan processing using
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
import time
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import Scan
from app.services.ai_tasks import claim_scan_for_processing, ScanClaimResult


def _create_scan(db: Session, scan_id: str, status: str = "PENDING") -> Scan:
    """Helper to create a scan record with given status."""
    scan = Scan(id=scan_id, filename=f"{scan_id}/source_study", status=status)
    db.add(scan)
    db.commit()
    return scan


def _get_scan_status(db: Session, scan_id: str) -> str | None:
    """Helper to get current scan status."""
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    return scan.status if scan else None


class TestScanClaim:
    """Test the scan claim/locking mechanism."""

    def test_claim_pending_succeeds(self, db_session: Session):
        """PENDING scan can be claimed successfully."""
        scan_id = "test-claim-pending-001"
        _create_scan(db_session, scan_id, "PENDING")
        
        result = claim_scan_for_processing(db_session, scan_id)
        
        assert result.success is True
        assert result.reason == "CLAIMED"
        assert result.scan_id == scan_id
        assert _get_scan_status(db_session, scan_id) == "PROCESSING"

    def test_claim_failed_succeeds(self, db_session: Session):
        """FAILED scan can be claimed (retry)."""
        scan_id = "test-claim-failed-002"
        _create_scan(db_session, scan_id, "FAILED")
        
        result = claim_scan_for_processing(db_session, scan_id)
        
        assert result.success is True
        assert result.reason == "CLAIMED"
        assert result.scan_id == scan_id
        assert _get_scan_status(db_session, scan_id) == "PROCESSING"

    def test_claim_processing_rejected(self, db_session: Session):
        """PROCESSING scan cannot be claimed by another worker."""
        scan_id = "test-claim-processing-003"
        _create_scan(db_session, scan_id, "PROCESSING")
        
        result = claim_scan_for_processing(db_session, scan_id)
        
        assert result.success is False
        assert result.reason == "ALREADY_PROCESSING"
        assert result.scan_id == scan_id
        # Status should remain PROCESSING
        assert _get_scan_status(db_session, scan_id) == "PROCESSING"

    def test_claim_segmented_rejected(self, db_session: Session):
        """SEGMENTED scan cannot be claimed (already complete)."""
        scan_id = "test-claim-segmented-004"
        _create_scan(db_session, scan_id, "SEGMENTED")
        
        result = claim_scan_for_processing(db_session, scan_id)
        
        assert result.success is False
        assert result.reason == "ALREADY_COMPLETE"
        assert result.scan_id == scan_id
        # Status should remain SEGMENTED
        assert _get_scan_status(db_session, scan_id) == "SEGMENTED"

    def test_claim_not_found_rejected(self, db_session: Session):
        """Non-existent scan returns NOT_FOUND."""
        scan_id = "test-claim-notfound-005"
        
        result = claim_scan_for_processing(db_session, scan_id)
        
        assert result.success is False
        assert result.reason == "NOT_FOUND"
        assert result.scan_id == scan_id

    def test_concurrent_claim_only_one_succeeds(self, db_session: Session):
        """Two concurrent claim attempts: exactly one succeeds."""
        scan_id = "test-claim-concurrent-006"
        _create_scan(db_session, scan_id, "PENDING")
        
        results = []
        barrier = threading.Barrier(2)
        
        def attempt_claim():
            # Each thread needs its own session
            local_db = SessionLocal()
            try:
                barrier.wait()  # Synchronize start
                result = claim_scan_for_processing(local_db, scan_id)
                results.append(result)
                if result.success:
                    local_db.commit()
                else:
                    local_db.rollback()
            finally:
                local_db.close()
        
        t1 = threading.Thread(target=attempt_claim)
        t2 = threading.Thread(target=attempt_claim)
        
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        # Exactly one should succeed
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        
        assert len(successes) == 1, f"Expected 1 success, got {len(successes)}"
        assert len(failures) == 1, f"Expected 1 failure, got {len(failures)}"
        assert failures[0].reason == "ALREADY_PROCESSING"
        assert _get_scan_status(db_session, scan_id) == "PROCESSING"

    def test_claim_sets_processing_started_at(self, db_session: Session):
        """Claimed scan has processing_started_at set."""
        scan_id = "test-claim-timestamp-007"
        _create_scan(db_session, scan_id, "PENDING")
        
        result = claim_scan_for_processing(db_session, scan_id)
        
        assert result.success is True
        
        # Verify processing_started_at was set
        scan = db_session.query(Scan).filter(Scan.id == scan_id).first()
        assert scan.processing_started_at is not None
        assert scan.status == "PROCESSING"


# Pytest fixture for database session
@pytest.fixture
def db_session() -> Session:
    """Provide a clean database session for each test."""
    db = SessionLocal()
    try:
        # Clean up any existing test scans
        db.query(Scan).filter(Scan.id.like("test-claim-%")).delete()
        db.commit()
        yield db
    finally:
        # Cleanup after test
        db.query(Scan).filter(Scan.id.like("test-claim-%")).delete()
        db.commit()
        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])