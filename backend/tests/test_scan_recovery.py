"""
Stale-PENDING scan recovery loop tests.

Tests verify the atomic, durable retry accounting for re-dispatching stranded
PENDING scans, the maximum-of-3-attempts cap, concurrency guarantees, and
cycle-level failure resilience.

Requires a live PostgreSQL (row-level locking via SELECT ... FOR UPDATE NOWAIT);
these are never run on SQLite and are excluded from the non-database suite.
"""
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

# Set required environment variables BEFORE importing app modules
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core")
os.environ.setdefault("MINIO_ACCESS_KEY", "testaccess")
os.environ.setdefault("MINIO_SECRET_KEY", "testsecret")

import pytest
import threading
from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import Scan, ModalityFile
from app.services.ai_tasks import (
    recover_stale_pending_scans,
    process_scan_task,
    claim_scan_for_processing,
    MAX_RECOVERY_ATTEMPTS,
    STALE_PENDING_THRESHOLD_SECONDS,
    _recovery_cycle,
)


def _create_scan(
    db: Session,
    scan_id: str,
    status: str = "PENDING",
    created_at: datetime | None = None,
    attempt_count: int = 0,
    processing_started_at: datetime | None = None,
) -> Scan:
    """Helper to create a scan record with given attributes."""
    now = datetime.now(timezone.utc)
    scan = Scan(
        id=scan_id,
        filename=f"{scan_id}/source_study",
        status=status,
        created_at=created_at or now,
        attempt_count=attempt_count,
        processing_started_at=processing_started_at,
    )
    db.add(scan)
    db.commit()
    return scan


def _create_modality_files(db: Session, scan_id: str) -> dict[str, str]:
    """Helper to create modality file records for a scan."""
    modality_objects: dict[str, str] = {}
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


def _get_scan(db: Session, scan_id: str) -> Scan | None:
    return db.query(Scan).filter(Scan.id == scan_id).first()


def _stale_created_at() -> datetime:
    """A created_at older than the staleness threshold."""
    return datetime.now(timezone.utc) - timedelta(
        seconds=STALE_PENDING_THRESHOLD_SECONDS + 60
    )


def _recent_created_at() -> datetime:
    """A created_at within the staleness threshold (fresh upload)."""
    return datetime.now(timezone.utc)


class TestStalePendingRecovery:
    """Test the stale-PENDING recovery accounting and claim."""

    def test_recover_stale_pending_increments_attempt_and_claims(
        self, db_session: Session
    ):
        """A stale PENDING scan is claimed (PROCESSING) and attempt_count=1."""
        scan_id = "test-recovery-stale-001"
        _create_scan(db_session, scan_id, "PENDING", created_at=_stale_created_at())
        _create_modality_files(db_session, scan_id)

        pairs = recover_stale_pending_scans(db_session)

        assert len(pairs) == 1
        returned_scan_id, modality_objects, skip_claim = pairs[0]
        assert returned_scan_id == scan_id
        assert skip_claim is True
        assert set(modality_objects) == {"t1", "t1ce", "t2", "flair"}

        scan = _get_scan(db_session, scan_id)
        assert scan is not None
        assert scan.status == "PROCESSING"
        assert scan.attempt_count == 1
        assert scan.processing_started_at is not None

    def test_recover_skips_recent_pending_scan(self, db_session: Session):
        """A recently-uploaded PENDING scan is not considered stale."""
        scan_id = "test-recovery-recent-002"
        _create_scan(db_session, scan_id, "PENDING", created_at=_recent_created_at())
        _create_modality_files(db_session, scan_id)

        pairs = recover_stale_pending_scans(db_session)

        assert pairs == []
        scan = _get_scan(db_session, scan_id)
        assert scan.status == "PENDING"
        assert scan.attempt_count == 0

    def test_recover_skips_processing_scan(self, db_session: Session):
        """PROCESSING scans are never reclaimed; attempt_count unchanged."""
        scan_id = "test-recovery-processing-003"
        _create_scan(
            db_session,
            scan_id,
            "PROCESSING",
            created_at=_stale_created_at(),
            attempt_count=0,
        )
        _create_modality_files(db_session, scan_id)

        pairs = recover_stale_pending_scans(db_session)

        assert pairs == []
        scan = _get_scan(db_session, scan_id)
        assert scan.status == "PROCESSING"
        assert scan.attempt_count == 0

    def test_recover_skips_segmented_and_failed_scans(self, db_session: Session):
        """SEGMENTED and FAILED scans are not recovered."""
        segmented_id = "test-recovery-seg-004"
        failed_id = "test-recovery-failed-004"
        _create_scan(db_session, segmented_id, "SEGMENTED", created_at=_stale_created_at())
        _create_scan(db_session, failed_id, "FAILED", created_at=_stale_created_at())
        _create_modality_files(db_session, segmented_id)
        _create_modality_files(db_session, failed_id)

        pairs = recover_stale_pending_scans(db_session)

        assert pairs == []
        assert _get_scan(db_session, segmented_id).status == "SEGMENTED"
        assert _get_scan(db_session, failed_id).status == "FAILED"

    def test_recovery_max_attempts_cap_marks_failed(self, db_session: Session):
        """After MAX_RECOVERY_ATTEMPTS the scan is marked FAILED and not retried."""
        scan_id = "test-recovery-cap-005"
        _create_scan(
            db_session,
            scan_id,
            "PENDING",
            created_at=_stale_created_at(),
            attempt_count=MAX_RECOVERY_ATTEMPTS - 1,  # 2
        )
        _create_modality_files(db_session, scan_id)

        # Third attempt (2 -> 3): claimed for dispatch, not yet at cap.
        pairs = recover_stale_pending_scans(db_session)
        assert len(pairs) == 1
        scan = _get_scan(db_session, scan_id)
        assert scan.status == "PROCESSING"
        assert scan.attempt_count == MAX_RECOVERY_ATTEMPTS  # 3

        # Reset to PENDING at the cap to simulate another stranded recovery.
        scan.status = "PENDING"
        db_session.commit()

        # Fourth attempt would exceed the cap -> marked FAILED, not returned.
        pairs = recover_stale_pending_scans(db_session)
        assert pairs == []
        scan = _get_scan(db_session, scan_id)
        assert scan.status == "FAILED"
        assert scan.attempt_count == MAX_RECOVERY_ATTEMPTS

        # A FAILED scan is not retried by subsequent cycles.
        pairs = recover_stale_pending_scans(db_session)
        assert pairs == []
        assert _get_scan(db_session, scan_id).status == "FAILED"

    def test_recovery_concurrent_no_duplicate_attempt(self, db_session: Session):
        """Two concurrent recovery cycles increment attempt_count exactly once."""
        scan_id = "test-recovery-concurrent-006"
        _create_scan(db_session, scan_id, "PENDING", created_at=_stale_created_at())
        _create_modality_files(db_session, scan_id)

        results: list[list] = []
        errors: list[Exception] = []

        def run_recovery():
            local_db = SessionLocal()
            try:
                pairs = recover_stale_pending_scans(local_db)
                results.append(pairs)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                local_db.close()

        t1 = threading.Thread(target=run_recovery)
        t2 = threading.Thread(target=run_recovery)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Concurrent recovery raised: {errors}"

        # Read through a fresh session: db_session's identity cache holds the
        # pre-recovery state because the threads committed via separate sessions.
        fresh = SessionLocal()
        try:
            scan = fresh.query(Scan).filter(Scan.id == scan_id).one()
        finally:
            fresh.close()
        # Exactly one cycle claimed the scan and incremented attempt_count once.
        assert scan.attempt_count == 1, f"Expected attempt_count 1, got {scan.attempt_count}"
        assert scan.status == "PROCESSING"
        # Exactly one cycle returned the scan for dispatch.
        returned = [p for pairs in results for p in pairs]
        assert len(returned) == 1, f"Expected 1 dispatch pair, got {len(returned)}"

    def test_recovery_claim_path_preserved_for_upload(self, db_session: Session):
        """process_scan_task with skip_claim=False performs the atomic claim itself;
        skip_claim=True (recovery) skips it because recovery already claimed."""
        scan_id = "test-recovery-claimpath-007"
        _create_scan(db_session, scan_id, "PENDING", created_at=_recent_created_at())

        with patch(
            "app.services.ai_tasks.claim_scan_for_processing",
            wraps=claim_scan_for_processing,
        ) as mock_claim, patch(
            "app.services.ai_tasks.preprocess_brats_study",
            side_effect=Exception("stop after claim"),
        ), patch(
            "app.services.ai_tasks.minio_client"
        ):
            # skip_claim=False (upload path): claim is performed.
            try:
                process_scan_task(scan_id, {"t1": "x.nii.gz"}, skip_claim=False)
            except Exception:
                pass
            assert mock_claim.call_count == 1

            # The claim transitioned PENDING -> PROCESSING (committed by
            # process_scan_task's own session; read via a fresh session to
            # avoid db_session's identity cache).
            fresh = SessionLocal()
            try:
                scan = fresh.query(Scan).filter(Scan.id == scan_id).one()
            finally:
                fresh.close()
            assert scan.status == "PROCESSING"
            assert scan.attempt_count == 0  # upload path does not count recovery attempts


class TestRecoveryCycle:
    """Test the async recovery cycle: dispatch off event loop, continue after failures."""

    @pytest.mark.asyncio
    async def test_recovery_cycle_dispatches_stale_scan(self, db_session: Session):
        """_recovery_cycle recovers a stale PENDING scan and returns it for dispatch."""
        scan_id = "test-recovery-cycle-dispatch-008"
        _create_scan(db_session, scan_id, "PENDING", created_at=_stale_created_at())
        _create_modality_files(db_session, scan_id)

        with patch(
            "app.services.ai_tasks.process_scan_task", return_value=None
        ) as mock_process:
            pairs = await _recovery_cycle(staleness_seconds=1)

        assert len(pairs) == 1
        assert pairs[0][0] == scan_id
        assert pairs[0][2] is True  # skip_claim
        mock_process.assert_called_once_with(scan_id, pairs[0][1], True)

    @pytest.mark.asyncio
    async def test_recovery_cycle_continues_after_dispatch_failure(
        self, db_session: Session
    ):
        """A failed dispatch is logged and does not abort the recovery cycle."""
        scan_id = "test-recovery-cycle-fail-009"
        _create_scan(db_session, scan_id, "PENDING", created_at=_stale_created_at())
        _create_modality_files(db_session, scan_id)

        with patch(
            "app.services.ai_tasks.process_scan_task",
            side_effect=Exception("Simulated inference failure"),
        ) as mock_process:
            # Should not raise; cycle continues despite dispatch failure.
            pairs = await _recovery_cycle(staleness_seconds=1)

        # Recovery accounting still happened (claimed + attempt counted).
        assert len(pairs) == 1
        assert pairs[0][0] == scan_id
        mock_process.assert_called_once()

        scan = _get_scan(db_session, scan_id)
        assert scan.status == "PROCESSING"
        assert scan.attempt_count == 1

    @pytest.mark.asyncio
    async def test_recovery_cycle_no_stale_scans(self, db_session: Session):
        """A cycle with no stale PENDING scans returns an empty list."""
        pairs = await _recovery_cycle(staleness_seconds=1)
        assert pairs == []


# Pytest fixture for database session
@pytest.fixture
def db_session() -> Session:
    """Provide a clean database session for each test."""
    db = SessionLocal()
    try:
        db.query(Scan).filter(Scan.id.like("test-recovery-%")).delete()
        db.commit()
        yield db
    finally:
        db.query(Scan).filter(Scan.id.like("test-recovery-%")).delete()
        db.query(ModalityFile).filter(ModalityFile.scan_id.like("test-recovery-%")).delete()
        db.commit()
        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
