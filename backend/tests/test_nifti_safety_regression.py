"""
SG3-F3 NIfTI Processing Safety Regression Tests

Focused regression tests for the NIfTI validation added by the SG3-F3 batch
(upload validation, integrity validation, resource bounds, and the worker-side
safety fix), extended by SG3-F3-I (bounded upload validation and float32-aware
resource bounds).

Scope of this module:
    * upload validation of ``.nii`` / ``.nii.gz`` (magic at the real header
      offset, dimensions/datatype/voxel-count/``vox_offset``/payload-size
      consistency, bounded streaming gzip with truncation, CRC, concatenated
      member, and trailing-garbage rejection);
    * header validation against malformed dimensions, unsupported datatypes,
      NIfTI magic, and non-NIfTI objects;
    * validation ordering: header metadata is checked *before* the volumetric
      payload is materialised (``get_fdata``) and *before* inference runs;
    * a header-only validator does not inspect the data extent (preprocessing),
      while the upload validator rejects truncated payloads up front;
    * training / evaluation segmentation validation before full allocation;
    * a valid supported NIfTI file still loads.

These tests use tiny synthetic fixtures (a few hundred bytes / a few KB) and
mocks only. No real patient data and no large volumes are created.

Run with: python -m pytest backend/tests/test_nifti_safety_regression.py -v
"""

import gzip
import io
import os
import struct
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
from fastapi import HTTPException

from ai_pipeline.preprocessing import _validate_nifti_image
from ai_pipeline.preprocessing.brats_preprocessing import (
    BraTSPreprocessor,
    PreprocessingConfig,
    MAX_VOLUME_DIMENSION,
    SUPPORTED_DTYPES,
    _validate_nifti_shape_and_dtype,
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
from app.api.routers import scans

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


# NIfTI-1 header field offsets (within the 348-byte header).
_H_DIM = 40
_H_DATATYPE = 70
_H_BITPIX = 72
_H_PIXDIM = 76
_H_VOX_OFFSET = 108
_H_MAGIC = 344

# datatype code for each supported numpy dtype (inverse of NIFTI1_DTYPE_CODES)
_DTYPE_CODES = {
    np.dtype(np.uint8): 2,
    np.dtype(np.int16): 4,
    np.dtype(np.int32): 8,
    np.dtype(np.float32): 16,
    np.dtype(np.float64): 64,
    np.dtype(np.uint16): 512,
}


def _tiny_nii_bytes(shape=TINY_SHAPE, dtype=np.float32) -> bytes:
    """Return the raw bytes of a tiny valid single-file NIfTI-1 (.nii)."""
    staging = Path(tempfile.mkdtemp()) / "tiny.nii"
    _save_nifti(staging, shape=shape, dtype=dtype)
    data = staging.read_bytes()
    staging.unlink()
    return data


def _craft_header(shape, dtype=np.float32, vox_offset=352.0, magic=b"n+1",
                  datatype_code=None, bitpix=None) -> bytes:
    """
    Build a 348-byte NIfTI-1 header declaring an arbitrary shape/dtype.

    Used to exercise the resource bounds without allocating a real volume.
    """
    dtype = np.dtype(dtype)
    code = _DTYPE_CODES.get(dtype) if datatype_code is None else datatype_code
    bits = dtype.itemsize * 8 if bitpix is None else bitpix
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)  # sizeof_hdr
    dims = list(shape) + [0] * (7 - len(shape))
    struct.pack_into("<8h", header, _H_DIM, len(shape), *dims)
    struct.pack_into("<h", header, _H_DATATYPE, code)
    struct.pack_into("<h", header, _H_BITPIX, bits)
    for i in range(1, 4):  # nonzero pixdim silences a nibabel warning
        struct.pack_into("<f", header, _H_PIXDIM + 4 * i, 1.0)
    struct.pack_into("<f", header, _H_VOX_OFFSET, vox_offset)
    header[_H_MAGIC:_H_MAGIC + 4] = magic + b"\x00"
    return bytes(header)


# Optional real-data compatibility check (read-only; header bytes only). The
# BraTS2020 sample is not part of the repository, so the test is skipped when it
# is absent. It never reads voxel data and never exposes patient data.
_REAL_BRATS_NII = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "BraTS_data"
    / "BraTS2020_TrainingData"
    / "BraTS20_Training_001"
    / "BraTS20_Training_001_t1.nii"
)


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
# Upload validation (bounded streaming; real validator functions)
# ---------------------------------------------------------------------------

class TestUploadNiftiValidation:
    """
    Exercise the real upload validator (``app.api.routers.scans``) with tiny
    fixtures. No large files, no network access, no temporary volumes.
    """

    def test_valid_uncompressed_nii_accepted(self):
        assert scans._validate_nifti_header(io.BytesIO(_tiny_nii_bytes()), ".nii") is True

    def test_valid_gzipped_nii_accepted(self):
        payload = gzip.compress(_tiny_nii_bytes())
        assert scans._validate_nifti_header(io.BytesIO(payload), ".nii.gz") is True

    def test_truncated_payload_rejected(self):
        truncated = _tiny_nii_bytes()[: 352 + 8]
        assert scans._validate_nifti_header(io.BytesIO(truncated), ".nii") is False

    def test_truncated_gzip_stream_rejected(self):
        payload = gzip.compress(_tiny_nii_bytes())
        truncated = payload[: len(payload) // 2]
        assert scans._validate_nifti_header(io.BytesIO(truncated), ".nii.gz") is False

    def test_crc_failure_rejected(self):
        payload = gzip.compress(_tiny_nii_bytes())
        corrupt = payload[:-8] + b"\x00" * 8  # clobber the CRC32/ISIZE trailer
        assert scans._validate_nifti_header(io.BytesIO(corrupt), ".nii.gz") is False

    def test_malformed_gzip_rejected(self):
        assert (
            scans._validate_nifti_header(io.BytesIO(b"\x1f\x8bnot-a-gzip-stream"), ".nii.gz")
            is False
        )

    def test_trailing_garbage_rejected(self):
        payload = gzip.compress(_tiny_nii_bytes())
        assert scans._validate_nifti_header(io.BytesIO(payload + b"garbage"), ".nii.gz") is False

    def test_concatenated_members_rejected(self):
        member = gzip.compress(_tiny_nii_bytes())
        assert scans._validate_nifti_header(io.BytesIO(member + member), ".nii.gz") is False

    def test_decompressed_size_limit_enforced_without_large_files(self, monkeypatch):
        """A tiny compressed stream that expands past the cap is rejected."""
        monkeypatch.setattr(scans, "MAX_DECOMPRESSED_BYTES", 1024 * 1024)
        bomb = gzip.compress(b"\x00" * (2 * 1024 * 1024))
        assert len(bomb) < 1024 * 1024  # the fixture itself is small
        assert scans._validate_nifti_header(io.BytesIO(bomb), ".nii.gz") is False

    def test_invalid_magic_rejected(self):
        raw = _tiny_nii_bytes()
        bad = raw[:344] + b"n+2\x00" + raw[348:]  # NIfTI-2 magic, not accepted
        assert scans._validate_nifti_header(io.BytesIO(bad), ".nii") is False

    def test_unsupported_datatype_rejected(self):
        header = _craft_header(TINY_SHAPE, np.int64, datatype_code=1024, bitpix=64)
        assert scans._validate_nifti_header(io.BytesIO(header), ".nii") is False

    def test_inconsistent_payload_metadata_rejected(self):
        # An extra trailing byte means the declared payload no longer matches.
        raw = _tiny_nii_bytes()
        assert scans._validate_nifti_header(io.BytesIO(raw + b"\x00"), ".nii") is False

    def test_extension_container_mismatch_rejected(self):
        gzipped = gzip.compress(_tiny_nii_bytes())
        assert scans._validate_nifti_header(io.BytesIO(gzipped), ".nii") is False
        assert scans._validate_nifti_header(io.BytesIO(_tiny_nii_bytes()), ".nii.gz") is False

    def test_excessive_4d_dimensions_rejected(self):
        header = _craft_header((100, 100, 100, 100), np.int16)
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            scans._validate_nifti1_single_file(header, 0)

    def test_extra_axis_cannot_inflate_the_estimate(self):
        # Any dimension above the cap is rejected...
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            _validate_nifti_shape_and_dtype(
                (4, 4, 4, MAX_VOLUME_DIMENSION + 1), np.dtype(np.uint8)
            )
        # ...and an extra axis is now included in the float32 allocation
        # estimate (previously only shape[:3] was counted, so this passed).
        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            _validate_nifti_shape_and_dtype((300, 300, 300, 2), np.dtype(np.float32))

    @pytest.mark.skipif(not _REAL_BRATS_NII.exists(), reason="BraTS2020 sample not present")
    def test_real_brats_uncompressed_sample_accepted(self):
        """
        A real BraTS2020 ``.nii`` (NIfTI-1, ``vox_offset`` != 352) is accepted.

        BraTS2020 ships uncompressed single-file NIfTI-1 volumes whose
        ``vox_offset`` accounts for header extensions (e.g. 2880), so this
        guards the offset-0 magic bug and the "don't assume 352" payload rule.
        Only the header is read; no voxel data is loaded.
        """
        with open(_REAL_BRATS_NII, "rb") as handle:
            assert scans._validate_nifti_header(handle, ".nii") is True


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
# Preprocessing resource bounds: every dimension is bounded, float32-aware
# ---------------------------------------------------------------------------

class TestPreprocessingResourceBounds:
    """Extra/oversized axes are rejected before any payload allocation."""

    @staticmethod
    def _load_crafted(tmp_path, shape, dtype=np.int16):
        path = tmp_path / "crafted.nii"
        path.write_bytes(_craft_header(shape, dtype))
        return nib.load(str(path))

    def test_4d_oversized_rejected_before_allocation(self, tmp_path, monkeypatch):
        image = self._load_crafted(tmp_path, (100, 100, 100, 100))

        calls = []
        original = nib.Nifti1Image.get_fdata

        def spy(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(nib.Nifti1Image, "get_fdata", spy)

        with pytest.raises(ValueError, match="exceeds maximum allowed"):
            _validate_nifti_image(image)

        assert calls == [], "validation must reject before materialising the volume"

    def test_5d_oversized_rejected(self, tmp_path):
        image = self._load_crafted(tmp_path, (100, 100, 100, 100, 2))
        with pytest.raises(ValueError, match="voxel count"):
            _validate_nifti_image(image)

    def test_moderate_4d_within_bounds_accepted(self, tmp_path):
        image = self._load_crafted(tmp_path, (60, 60, 60, 4))
        _validate_nifti_image(image)  # must not raise


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

    def test_oversized_4d_reference_image_rejected_before_inference(
        self, tmp_path, worker_env
    ):
        """An oversized 4D reference modality fails before inference runs."""
        install_fget, model_calls = worker_env

        crafted = gzip.compress(_craft_header((100, 100, 100, 100), np.int16))

        def writer(modality, path):
            if modality == "flair":
                Path(path).write_bytes(crafted)
            else:
                _save_nifti(path, shape=TINY_SHAPE, dtype=np.float32)

        install_fget(writer)

        modality_objects = {
            m: f"test-nifti-4d/source_{m}.nii.gz"
            for m in ("t1", "t1ce", "t2", "flair")
        }
        process_scan_task("test-nifti-4d", modality_objects)

        assert model_calls == [], "inference must not run for an oversized 4D image"

    def test_uncompressed_modality_suffix_preserved(self, worker_env):
        """
        The worker downloads an uncompressed study with its real ``.nii`` name
        (so both containers are loadable), and rejects before inference.
        """
        install_fget, model_calls = worker_env

        written = {}

        def writer(modality, path):
            written[modality] = Path(path).name
            if modality == "flair":
                Path(path).write_bytes(b"not a nifti")  # stop before inference
            else:
                _save_nifti(path, shape=TINY_SHAPE, dtype=np.float32)

        install_fget(writer)

        modality_objects = {
            m: f"test-nifti-plain/source_{m}.nii"
            for m in ("t1", "t1ce", "t2", "flair")
        }
        process_scan_task("test-nifti-plain", modality_objects)

        assert written == {
            "t1": "t1.nii",
            "t1ce": "t1ce.nii",
            "t2": "t2.nii",
            "flair": "flair.nii",
        }
        assert model_calls == []


# ---------------------------------------------------------------------------
# Uncompressed modality support (the preprocessor finds .nii names)
# ---------------------------------------------------------------------------

class TestUncompressedModalitySupport:
    """Both ``.nii`` and ``.nii.gz`` names resolve through the preprocessor."""

    def test_find_modality_file_matches_uncompressed(self, tmp_path):
        (tmp_path / "study_t1.nii").write_bytes(b"x")
        preprocessor = BraTSPreprocessor(PreprocessingConfig(image_size=224))
        assert preprocessor.find_modality_file(tmp_path, "t1").name == "study_t1.nii"

    def test_find_modality_file_matches_compressed(self, tmp_path):
        (tmp_path / "study_t1.nii.gz").write_bytes(b"x")
        preprocessor = BraTSPreprocessor(PreprocessingConfig(image_size=224))
        assert preprocessor.find_modality_file(tmp_path, "t1").name == "study_t1.nii.gz"


# ---------------------------------------------------------------------------
# Endpoint-level: dispatch on success, cleanup on rejection
# ---------------------------------------------------------------------------

class TestUploadEndpointCleanup:
    """
    Exercise the real ``upload_scan`` endpoint with mocked DB/MinIO: a valid
    study dispatches, and an invalid modality rejects the study while cleaning
    up the previously uploaded objects and leaving no temp files behind.
    """

    @staticmethod
    def _user():
        user = MagicMock()
        user.firebase_uid = "user-a-uid"
        return user

    @staticmethod
    def _upload(name, content):
        upload = MagicMock()
        upload.filename = name
        upload.file = io.BytesIO(content)
        return upload

    @staticmethod
    def _fake_minio():
        fake = MagicMock()
        fake.bucket_exists.return_value = True
        return fake

    def test_valid_uncompressed_study_dispatched(self, monkeypatch, tmp_path):
        import asyncio

        monkeypatch.setattr(scans.uuid, "uuid4", lambda: "ok-scan-id")
        monkeypatch.setattr(scans, "minio_client", self._fake_minio())
        monkeypatch.chdir(tmp_path)  # keep temp_uploads out of the repo tree

        raw = _tiny_nii_bytes(shape=(16, 16, 16))
        files = {m: self._upload(f"{m}.nii", raw) for m in ("t1", "t1ce", "t2", "flair")}

        result = asyncio.run(
            scans.upload_scan(
                background_tasks=MagicMock(),
                t1=files["t1"], t1ce=files["t1ce"], t2=files["t2"], flair=files["flair"],
                db=MagicMock(), current_user=self._user(),
            )
        )

        assert result.scan_id == "ok-scan-id"
        assert scans.minio_client.fput_object.call_count == 4
        uploaded = [c.args[1] for c in scans.minio_client.fput_object.call_args_list]
        assert all(name.endswith(".nii") for name in uploaded)
        assert scans.minio_client.remove_object.call_count == 0
        assert list(Path("temp_uploads").iterdir()) == []

    def test_invalid_modality_rejects_and_cleans_up(self, monkeypatch, tmp_path):
        import asyncio

        monkeypatch.setattr(scans.uuid, "uuid4", lambda: "cleanup-scan-id")
        monkeypatch.setattr(scans, "minio_client", self._fake_minio())
        monkeypatch.chdir(tmp_path)

        good = gzip.compress(_tiny_nii_bytes(shape=(16, 16, 16)))
        bad = gzip.compress(_tiny_nii_bytes())[:20]  # truncated gzip stream
        files = {
            "t1": self._upload("t1.nii.gz", good),
            "t1ce": self._upload("t1ce.nii.gz", good),
            "t2": self._upload("t2.nii.gz", bad),
            "flair": self._upload("flair.nii.gz", good),
        }

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                scans.upload_scan(
                    background_tasks=MagicMock(),
                    t1=files["t1"], t1ce=files["t1ce"], t2=files["t2"], flair=files["flair"],
                    db=MagicMock(), current_user=self._user(),
                )
            )

        assert exc_info.value.status_code == 400
        removed = [c.args[1] for c in scans.minio_client.remove_object.call_args_list]
        assert "cleanup-scan-id/source_t1.nii.gz" in removed
        assert "cleanup-scan-id/source_t1ce.nii.gz" in removed
        # The rejected modality was never uploaded, so it is not "cleaned up".
        assert "cleanup-scan-id/source_t2.nii.gz" not in removed
        # No partial temporary files remain for the rejected study.
        assert list(Path("temp_uploads").glob("cleanup-scan-id_*")) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
