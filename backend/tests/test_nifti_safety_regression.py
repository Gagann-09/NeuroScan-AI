"""
SG3-F3-G NIfTI Processing Safety Regression Tests

Focused regression tests for the NIfTI validation added by the SG3-F3 batch
(upload validation, integrity validation, resource bounds, and the worker-side
safety fix).

Scope of this module:
    * header validation against malformed dimensions, unsupported datatypes,
      NIfTI magic, and non-NIfTI objects;
    * validation ordering: header metadata is checked *before* the volumetric
      payload is materialised (``get_fdata``) and *before* inference runs;
    * truncated-payload behaviour (currently detected at read time, not by the
      header-only validator);
    * training / evaluation segmentation validation before full allocation;
    * a valid supported NIfTI file still loads.

These tests use tiny synthetic fixtures (a few hundred bytes) and mocks only.
No real patient data and no large volumes are created.

Run with: python -m pytest backend/tests/test_nifti_safety_regression.py -v
"""

import gzip
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Required environment variables must be present before importing app modules.
# These mirror the values used by the other app-importing test modules (e.g.
# test_auth.py). Using ``setdefault`` means whichever module is imported first
# wins, so the values must be identical across modules; in particular the
# explicit ``+psycopg2`` driver is required because the psycopg (v3) package is
# not installed in the test environment.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://neuroscan_admin:secure_password_123@localhost:5432/neuroscan_core",
)
os.environ.setdefault("MINIO_ACCESS_KEY", "testaccess")
os.environ.setdefault("MINIO_SECRET_KEY", "testsecret")
os.environ.setdefault("FIREBASE_PROJECT_ID", "neuroscan-medical-vault")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:3000")

import nibabel as nib
import numpy as np
import pytest

from ai_pipeline.preprocessing import _validate_nifti_image
from ai_pipeline.preprocessing.brats_preprocessing import (
    BraTSPreprocessor,
    PreprocessingConfig,
    MAX_VOLUME_DIMENSION,
    SUPPORTED_DTYPES,
)
from ai_pipeline.training.train_armt_gan import (
    BraTS2DSegmentationDataset,
    load_nifti as training_load_nifti,
)
from app.services.ai_tasks import (
    ScanClaimResult,
    _validate_nifti_file,
    process_scan_task,
)

# A deliberately tiny volume. 6*6*6 = 216 voxels (~864 bytes as float32).
TINY_SHAPE = (6, 6, 6)
NIFTI1_HEADER_BYTES = 352  # 348-byte header + 4-byte magic; data starts here.


# ---------------------------------------------------------------------------
# Fixture helpers (no large allocations)
# ---------------------------------------------------------------------------

def _save_nifti(path: Path, shape=TINY_SHAPE, dtype=np.float32) -> Path:
    """Save a tiny valid NIfTI-1 file and return its path."""
    path = Path(path)
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _int64_nifti_image(shape=TINY_SHAPE) -> nib.Nifti1Image:
    """Build an in-memory NIfTI-1 image whose stored datatype is int64."""
    data = np.zeros(shape, dtype=np.int64)
    header = nib.Nifti1Header()
    header.set_data_dtype(np.int64)
    header.set_data_shape(shape)
    return nib.Nifti1Image(data, np.eye(4), header=header)


def _save_truncated_gz(path: Path, shape=TINY_SHAPE) -> Path:
    """
    Write a gzipped NIfTI file whose header is intact but whose payload is
    truncated to 8 bytes (real payload is prod(shape)*4 bytes).
    """
    path = Path(path)
    staging = path.with_suffix(".staging.nii")
    _save_nifti(staging, shape=shape, dtype=np.float32)
    raw = staging.read_bytes()
    staging.unlink()
    truncated = raw[: NIFTI1_HEADER_BYTES + 8]
    path.write_bytes(gzip.compress(truncated))
    return path


def _modality_from_object_name(object_name: str) -> str:
    """Extract the modality from 'scan/source_<modality>.nii.gz'."""
    return object_name.split("source_")[1].split(".")[0]


# ---------------------------------------------------------------------------
# Header validation (validator level, no database)
# ---------------------------------------------------------------------------

class TestHeaderValidation:
    """Verify what the header validator does and does not protect against."""

    def test_supported_dtypes_are_dtype_instances(self):
        """Regression guard for the types-vs-dtype regression."""
        assert SUPPORTED_DTYPES, "SUPPORTED_DTYPES must not be empty"
        assert all(isinstance(dt, np.dtype) for dt in SUPPORTED_DTYPES), (
            "SUPPORTED_DTYPES must contain np.dtype instances so that set "
            "membership matches nibabel's get_data_dtype()"
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.int16, np.uint8])
    def test_valid_supported_dtypes_pass(self, dtype):
        """A valid NIfTI-1 file with a supported dtype passes validation."""
        data = np.zeros(TINY_SHAPE, dtype=dtype)
        image = nib.Nifti1Image(data, np.eye(4))
        _validate_nifti_image(image)  # must not raise

    def test_valid_nifti1_float32_loads_correctly(self, tmp_path):
        """A valid supported NIfTI file still loads correctly (end-to-end)."""
        path = _save_nifti(tmp_path / "study.nii.gz", dtype=np.float32)

        image = nib.load(str(path))
        _validate_nifti_file(path)

        volume = training_load_nifti(path)
        assert volume.dtype == np.float32
        assert volume.shape == TINY_SHAPE

        preprocessor = BraTSPreprocessor(PreprocessingConfig(image_size=224))
        volume2 = preprocessor.load_nifti(path)
        assert volume2.dtype == np.float32
        assert volume2.shape == TINY_SHAPE

    def test_unsupported_dtype_rejected(self):
        """An unsupported datatype (int64) is rejected."""
        image = _int64_nifti_image()
        assert image.get_data_dtype() == np.dtype(np.int64)
        with pytest.raises(ValueError, match="Unsupported data type"):
            _validate_nifti_image(image)

    def test_unsupported_dtype_rejected_on_disk(self, tmp_path):
        """The on-disk wrapper also rejects an unsupported datatype."""
        path = tmp_path / "seg_int64.nii.gz"
        nib.save(_int64_nifti_image(), str(path))
        with pytest.raises(ValueError, match="Unsupported data type"):
            _validate_nifti_file(path)

    def test_oversized_dimension_rejected(self):
        """A dimension larger than the volume cap is rejected in-memory (32KB)."""
        data = np.zeros((MAX_VOLUME_DIMENSION + 1, 4, 4), dtype=np.float32)
        image = nib.Nifti1Image(data, np.eye(4))
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            _validate_nifti_image(image)

    def test_two_dimensional_shape_rejected(self):
        """A shape with fewer than 3 dimensions is rejected."""
        image = nib.Nifti1Image(np.zeros((6, 6), dtype=np.float32), np.eye(4))
        assert len(image.shape) < 3
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            _validate_nifti_image(image)

    def test_invalid_magic_rejected(self):
        """A header whose magic is not ni1/n+1 is rejected."""
        image = nib.Nifti1Image(np.zeros(TINY_SHAPE, dtype=np.float32), np.eye(4))
        image.header["magic"] = b"n+2"  # NIfTI-2 magic, not in whitelist
        with pytest.raises(ValueError, match="Invalid NIfTI magic"):
            _validate_nifti_image(image)

    def test_non_nifti_object_rejected(self):
        """A non-NIfTI object is rejected as an unsupported image type."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            _validate_nifti_image(object())

    def test_validation_is_header_only_and_reads_no_payload(self, monkeypatch):
        """
        Validation must rely on header metadata only: get_fdata() must not be
        invoked. This is the "validate before full allocation" guarantee.
        """
        calls = []
        original = nib.Nifti1Image.get_fdata

        def spy(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(nib.Nifti1Image, "get_fdata", spy)

        image = nib.load(str(_save_nifti(Path(tempfile.mkdtemp()) / "ok.nii.gz")))
        _validate_nifti_image(image)

        assert calls == [], "validation must not materialise the volume data"

    def test_nifti2_rejected(self):
        """
        NIfTI-2 is not required by this project (BraTS is NIfTI-1). The
        validator documents NIfTI-1-only support, so a valid NIfTI-2 image must
        be rejected explicitly rather than silently accepted.
        """
        image = nib.Nifti2Image(np.zeros(TINY_SHAPE, dtype=np.float32), np.eye(4))
        assert image.header.get("magic") in (b"ni2", b"n+2")
        with pytest.raises(ValueError, match="NIfTI-1"):
            _validate_nifti_image(image)


# ---------------------------------------------------------------------------
# Truncated payloads
# ---------------------------------------------------------------------------

class TestTruncatedPayloads:
    """Truncation is not detected by header-only validation."""

    def test_truncated_payload_passes_header_validation(self, tmp_path):
        """
        Documents the current behaviour: header validation passes for a
        truncated payload because it never inspects the data extent.
        """
        path = _save_truncated_gz(tmp_path / "trunc_flair.nii.gz")
        image = nib.load(str(path))
        _validate_nifti_image(image)  # header-only: passes

    def test_truncated_payload_fails_on_read(self, tmp_path):
        """The truncated payload is rejected when the data is actually read."""
        path = _save_truncated_gz(tmp_path / "trunc_flair.nii.gz")
        image = nib.load(str(path))
        with pytest.raises(OSError):
            image.get_fdata()

    def test_truncated_file_rejected_by_disk_wrapper_only_at_read(self, tmp_path):
        """_validate_nifti_file is header-only; the error surfaces on read."""
        path = _save_truncated_gz(tmp_path / "trunc_t1.nii.gz")
        _validate_nifti_file(path)  # does not raise
        with pytest.raises(OSError):
            nib.load(str(path)).get_fdata()


# ---------------------------------------------------------------------------
# Training / evaluation segmentation validation before full allocation
# ---------------------------------------------------------------------------

class TestSegmentationValidationBeforeAllocation:
    """Invalid segmentation is rejected before the payload is allocated."""

    @staticmethod
    def _name_seg(patient_dir: Path) -> Path:
        return patient_dir / "patient_seg.nii.gz"

    def test_training_rejects_invalid_segmentation_dtype_before_allocation(
        self, tmp_path, monkeypatch
    ):
        path = self._name_seg(tmp_path)
        nib.save(_int64_nifti_image(), str(path))

        calls = []
        original = nib.Nifti1Image.get_fdata

        def spy(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(nib.Nifti1Image, "get_fdata", spy)

        with pytest.raises(ValueError, match="Unsupported data type"):
            BraTS2DSegmentationDataset(patient_dirs=[tmp_path], image_size=224)

        assert calls == [], "segmentation data was read before validation rejected it"

    def test_training_rejects_oversized_segmentation_before_allocation(
        self, tmp_path, monkeypatch
    ):
        path = self._name_seg(tmp_path)
        data = np.zeros((MAX_VOLUME_DIMENSION + 1, 4, 4), dtype=np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))

        calls = []
        original = nib.Nifti1Image.get_fdata

        def spy(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(nib.Nifti1Image, "get_fdata", spy)

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            BraTS2DSegmentationDataset(patient_dirs=[tmp_path], image_size=224)

        assert calls == []

    def test_evaluation_rejects_invalid_segmentation_dtype_before_allocation(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "patient_seg.nii.gz"
        nib.save(_int64_nifti_image(), str(path))

        calls = []
        original = nib.Nifti1Image.get_fdata

        def spy(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(nib.Nifti1Image, "get_fdata", spy)

        preprocessor = BraTSPreprocessor(PreprocessingConfig(image_size=224))
        with pytest.raises(ValueError, match="Unsupported data type"):
            preprocessor.load_nifti(path)

        assert calls == []

    def test_evaluation_rejects_oversized_segmentation_before_allocation(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "patient_seg.nii.gz"
        data = np.zeros((MAX_VOLUME_DIMENSION + 1, 4, 4), dtype=np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))

        calls = []
        original = nib.Nifti1Image.get_fdata

        def spy(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(nib.Nifti1Image, "get_fdata", spy)

        preprocessor = BraTSPreprocessor(PreprocessingConfig(image_size=224))
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            preprocessor.load_nifti(path)

        assert calls == []


# ---------------------------------------------------------------------------
# Worker-level: reference images rejected before inference
# ---------------------------------------------------------------------------

class TestReferenceImagesRejectedBeforeInference:
    """
    process_scan_task must reject invalid/truncated reference modalities before
    the ARMT-GAN model is loaded and inference runs.

    The database layer and MinIO are mocked; no live PostgreSQL is required.
    """

    @pytest.fixture
    def worker_env(self, monkeypatch):
        """Patch the worker's DB, MinIO, and model boundaries."""
        import app.services.ai_tasks as ai_tasks

        fake_db = MagicMock()
        monkeypatch.setattr(ai_tasks, "SessionLocal", lambda: fake_db)
        monkeypatch.setattr(
            ai_tasks,
            "claim_scan_for_processing",
            lambda db, scan_id: ScanClaimResult(
                success=True, reason="CLAIMED", scan_id=scan_id
            ),
        )

        model_calls = []
        monkeypatch.setattr(
            ai_tasks, "_get_global_model", lambda: model_calls.append(1)
        )

        fake_minio = MagicMock()
        monkeypatch.setattr(ai_tasks, "minio_client", fake_minio)

        def install_fget(writer_for_modality):
            def fget(bucket, object_name, local_path):
                modality = _modality_from_object_name(object_name)
                writer_for_modality(modality, Path(local_path))

            fake_minio.fget_object.side_effect = fget

        return install_fget, model_calls

    @staticmethod
    def _valid_writer(modality, path):
        _save_nifti(path, shape=TINY_SHAPE, dtype=np.float32)

    def test_invalid_reference_image_rejected_before_inference(
        self, tmp_path, worker_env
    ):
        """A reference modality with an unsupported dtype fails before inference."""
        install_fget, model_calls = worker_env

        def writer(modality, path):
            if modality == "flair":
                nib.save(_int64_nifti_image(), str(path))
            else:
                _save_nifti(path, shape=TINY_SHAPE, dtype=np.float32)

        install_fget(writer)

        modality_objects = {
            m: f"test-nifti-invalid/source_{m}.nii.gz"
            for m in ("t1", "t1ce", "t2", "flair")
        }
        process_scan_task("test-nifti-invalid", modality_objects)

        assert model_calls == [], "inference must not run for an invalid reference image"

    def test_truncated_reference_image_rejected_before_inference(
        self, tmp_path, worker_env
    ):
        """A truncated reference modality fails before inference runs."""
        install_fget, model_calls = worker_env

        def writer(modality, path):
            if modality == "flair":
                _save_truncated_gz(path)
            else:
                _save_nifti(path, shape=TINY_SHAPE, dtype=np.float32)

        install_fget(writer)

        modality_objects = {
            m: f"test-nifti-truncated/source_{m}.nii.gz"
            for m in ("t1", "t1ce", "t2", "flair")
        }
        process_scan_task("test-nifti-truncated", modality_objects)

        assert model_calls == [], "inference must not run for a truncated reference image"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
