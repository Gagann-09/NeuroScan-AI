"""
Scans API Router.
Handles upload, status polling, and results retrieval for brain MRI scans.
All endpoint logic that was previously in main.py is consolidated here.
"""
import os
import uuid
import shutil
import logging
import math
import struct
import zlib
from typing import List, BinaryIO

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Form
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.core.rate_limiting import enforce_upload_rate_limit
from app.db.models import Scan, ModalityFile, Prediction, User, KaggleScan, KagglePrediction, KaggleModelVersion
from app.core.storage import minio_client, get_presigned_url
from app.services.brats_metadata import resolve_brats2020_cohort
from app.services.ai_tasks import process_scan_task, process_kaggle_task
from app.schemas.scan_schema import UploadRequest, UploadResponse, StatusResponse, ResultsResponse, KaggleUploadResponse, KaggleStatusResponse, KaggleResultsResponse
from ai_pipeline.preprocessing import (
    MAX_DECOMPRESSED_SIZE_MB,
    NIFTI1_DTYPE_CODES,
    _validate_nifti_shape_and_dtype,
)

router = APIRouter(prefix="/api/v1/scans", tags=["scans"])

logger = logging.getLogger(__name__)

# Maximum file size per modality (bytes). BraTS volumes are ~36MB uncompressed,
# ~5-15MB compressed. Set generous limit to prevent abuse while allowing valid scans.
MAX_MODALITY_FILE_SIZE = 100 * 1024 * 1024  # 100 MB per modality
ALLOWED_EXTENSIONS = (".nii", ".nii.gz")

# Single-file NIfTI-1 magic (at header offset 344). b"ni1" is the (unsupported)
# two-file .hdr/.img pair magic. GZIP_MAGIC is the gzip container marker.
NIFTI1_SINGLE_MAGIC = b"n+1"
GZIP_MAGIC = b"\x1f\x8b"

# NIfTI-1 header layout (offsets within the 348-byte header). The NIfTI-2 layout
# differs and is intentionally not supported (BraTS is NIfTI-1).
NIFTI1_HEADER_BYTES = 348
NIFTI1_SIZE_OF_HDR = 348
NIFTI1_DIM_OFFSET = 40          # int16[8]: dim[0]=ndim, dim[1..7]=axis lengths
NIFTI1_DATATYPE_OFFSET = 70     # int16 datatype code
NIFTI1_BITPIX_OFFSET = 72       # int16 bits per voxel
NIFTI1_VOX_OFFSET_OFFSET = 108  # float32 offset to the first data byte
NIFTI1_MAGIC_OFFSET = 344       # 3-byte magic + NUL
# Smallest valid data offset for a single-file NIfTI-1 (348-byte header + the
# 4-byte extension marker). Real files may be larger (header extensions).
NIFTI1_MIN_DATA_OFFSET = 352

# True decompressed-payload bound per modality. gzip input is streamed and
# counted, so this is an actual byte budget, not a header-derived estimate.
# Defaults to the agreed 200 MiB limit (MAX_DECOMPRESSED_SIZE_MB).
MAX_DECOMPRESSED_BYTES = MAX_DECOMPRESSED_SIZE_MB * 1024 * 1024

# gzip is decompressed in bounded chunks; the whole payload is never buffered.
GZIP_CHUNK_BYTES = 64 * 1024


def _parse_nifti1_header(header: bytes):
    """
    Parse a NIfTI-1 header and enforce format-level consistency.

    Returns ``(magic, shape, dtype, bitpix, data_offset)`` where ``magic`` is the
    3-byte single/pair magic (e.g. ``b"n+1"``), ``shape`` is the declared data
    shape, ``dtype`` is the numpy dtype for the stored datatype code, and
    ``data_offset`` is the integer ``vox_offset``.

    Raises ``ValueError`` for anything that is not a well-formed NIfTI-1 header.
    """
    if len(header) < NIFTI1_HEADER_BYTES:
        raise ValueError("header shorter than 348 bytes")

    # NIfTI-1 may be little- or big-endian; sizeof_hdr disambiguates the two.
    (little_endian,) = struct.unpack_from("<i", header, 0)
    (big_endian,) = struct.unpack_from(">i", header, 0)
    if little_endian == NIFTI1_SIZE_OF_HDR:
        endian = "<"
    elif big_endian == NIFTI1_SIZE_OF_HDR:
        endian = ">"
    else:
        raise ValueError("sizeof_hdr is not 348 (not a NIfTI-1 single-file header)")

    magic = header[NIFTI1_MAGIC_OFFSET:NIFTI1_MAGIC_OFFSET + 3]

    dims = struct.unpack_from(endian + "8h", header, NIFTI1_DIM_OFFSET)
    ndim = dims[0]
    if not 1 <= ndim <= 7:
        raise ValueError(f"invalid NIfTI-1 dimension count: {ndim}")
    shape = tuple(int(d) for d in dims[1:1 + ndim])

    datatype = struct.unpack_from(endian + "h", header, NIFTI1_DATATYPE_OFFSET)[0]
    dtype = NIFTI1_DTYPE_CODES.get(datatype)
    if dtype is None:
        raise ValueError(f"unsupported NIfTI-1 datatype code: {datatype}")

    bitpix = struct.unpack_from(endian + "h", header, NIFTI1_BITPIX_OFFSET)[0]
    if bitpix != dtype.itemsize * 8:
        raise ValueError(f"inconsistent bitpix ({bitpix}) for datatype code {datatype}")

    vox_offset = struct.unpack_from(endian + "f", header, NIFTI1_VOX_OFFSET_OFFSET)[0]
    if not math.isfinite(vox_offset) or vox_offset != int(vox_offset):
        raise ValueError(f"invalid vox_offset: {vox_offset!r}")

    return magic, shape, dtype, bitpix, int(vox_offset)


def _validate_nifti1_single_file(header: bytes, total_bytes: int) -> None:
    """
    Validate a single-file NIfTI-1 payload against its header.

    ``total_bytes`` is the true payload size: the file size for uncompressed
    input, or the decompressed byte count for gzip input. Raises ``ValueError``
    on any inconsistency (format, bounds, or declared-vs-actual payload size).
    """
    magic, shape, dtype, bitpix, data_offset = _parse_nifti1_header(header)

    if magic != NIFTI1_SINGLE_MAGIC:
        raise ValueError(
            f"unsupported NIfTI-1 magic {magic!r}; expected single-file "
            f"{NIFTI1_SINGLE_MAGIC!r} (paired .hdr/.img uploads are not supported)"
        )

    # Shared scientific/resource bounds (datatype, every dimension, voxel count,
    # float32 allocation estimate).
    _validate_nifti_shape_and_dtype(shape, dtype)

    if data_offset < NIFTI1_MIN_DATA_OFFSET:
        raise ValueError(
            f"vox_offset ({data_offset}) is below the single-file minimum "
            f"({NIFTI1_MIN_DATA_OFFSET})"
        )

    voxel_count = 1
    for dim in shape:
        voxel_count *= dim
    expected_bytes = data_offset + voxel_count * (bitpix // 8)

    if total_bytes < expected_bytes:
        raise ValueError(
            f"truncated payload: header declares {expected_bytes} bytes, "
            f"only {total_bytes} present"
        )
    if total_bytes > expected_bytes:
        raise ValueError(
            f"trailing data: header declares {expected_bytes} bytes, "
            f"{total_bytes} present"
        )


def _decompress_gzip_bounded(stream: BinaryIO, max_bytes: int):
    """
    Stream-decompress a single-member gzip stream with a hard byte bound.

    Reads through EOF so the gzip CRC32/ISIZE integrity trailer is verified, and
    rejects malformed, truncated, concatenated, and trailing data. Returns
    ``(total_decompressed_bytes, first_header_bytes)``. Never buffers the whole
    payload and never writes temporary files.

    Raises ``ValueError`` on any failure.
    """
    decompressor = zlib.decompressobj(31)  # 16 (gzip wrapper) + 15 (window)
    total = 0
    header = bytearray()
    compressed = 0

    while True:
        chunk = stream.read(GZIP_CHUNK_BYTES)
        if not chunk:
            break
        compressed += len(chunk)
        if compressed > MAX_MODALITY_FILE_SIZE:
            raise ValueError(
                f"compressed payload exceeds {MAX_MODALITY_FILE_SIZE // (1024 * 1024)} MB limit"
            )
        try:
            out = decompressor.decompress(chunk)
        except zlib.error as exc:
            raise ValueError(f"malformed gzip stream: {exc}") from exc
        total += len(out)
        if total > max_bytes:
            raise ValueError(
                f"decompressed payload exceeds {max_bytes // (1024 * 1024)} MB limit"
            )
        if len(header) < NIFTI1_HEADER_BYTES:
            header.extend(out[: NIFTI1_HEADER_BYTES - len(header)])
        if decompressor.eof:
            # The first member ended. Anything that follows is a second member or
            # trailing garbage; both are rejected explicitly.
            if decompressor.unused_data or stream.read(1):
                raise ValueError(
                    "trailing data after gzip member "
                    "(concatenated members or trailing garbage are not allowed)"
                )
            break

    if not decompressor.eof:
        raise ValueError("truncated gzip stream (end-of-stream marker or checksum missing)")

    return total, bytes(header)


def _validate_uncompressed_nifti(file_obj: BinaryIO) -> bool:
    """Validate an uncompressed single-file NIfTI-1 upload."""
    file_obj.seek(0, os.SEEK_END)
    total_bytes = file_obj.tell()
    file_obj.seek(0)
    if total_bytes > MAX_MODALITY_FILE_SIZE:
        return False
    header = file_obj.read(NIFTI1_HEADER_BYTES)
    try:
        _validate_nifti1_single_file(header, total_bytes)
    except ValueError:
        return False
    return True


def _validate_gzipped_nifti(file_obj: BinaryIO) -> bool:
    """Validate a gzipped single-file NIfTI-1 upload (bounded streaming)."""
    file_obj.seek(0)
    try:
        total_bytes, header = _decompress_gzip_bounded(file_obj, MAX_DECOMPRESSED_BYTES)
        _validate_nifti1_single_file(header, total_bytes)
    except ValueError:
        return False
    return True


def _validate_nifti_header(file_obj: BinaryIO, file_ext: str) -> bool:
    """
    Validate a single-file NIfTI-1 upload (``.nii`` or ``.nii.gz``).

    The container is detected from its leading magic and must match the declared
    extension. Uncompressed input is size-checked directly; gzip input is
    decompressed in bounded chunks through EOF (verifying the gzip checksum)
    under a hard decompressed-byte cap. The NIfTI-1 magic is read at its real
    header offset (344), and dimensions, datatype, voxel count, ``vox_offset``,
    and declared-vs-actual payload size are all checked before acceptance.
    """
    original_pos = file_obj.tell()
    try:
        file_obj.seek(0)
        marker = file_obj.read(2)
        file_obj.seek(0)
        if len(marker) < 2:
            return False

        is_gzip = marker == GZIP_MAGIC
        # The container must agree with the declared extension; the worker stores
        # the file under the same extension it was uploaded with.
        if file_ext == ".nii.gz" and not is_gzip:
            return False
        if file_ext == ".nii" and is_gzip:
            return False

        if is_gzip:
            return _validate_gzipped_nifti(file_obj)
        return _validate_uncompressed_nifti(file_obj)
    except Exception:
        return False
    finally:
        file_obj.seek(original_pos)


def _copyfileobj_with_limit(src: BinaryIO, dst: BinaryIO, limit: int) -> int:
    """
    Copy data from src to dst while enforcing a byte limit.
    Raises HTTPException if limit exceeded.
    Returns total bytes copied.
    """
    total = 0
    chunk_size = 8192  # 8KB chunks
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {limit // (1024*1024)} MB"
            )
        dst.write(chunk)
    return total


def _cleanup_source_objects(bucket_name: str, object_names: List[str]) -> None:
    """
    Best-effort cleanup of source objects from MinIO.
    Cleanup failures are logged but do not raise; original errors must propagate.
    Already-missing objects are treated as successfully cleaned up.
    """
    for obj_name in object_names:
        try:
            minio_client.remove_object(bucket_name, obj_name)
        except Exception as e:
            # Check if object already doesn't exist (S3 error code NoSuchKey)
            error_code = getattr(e, "code", None)
            if error_code == "NoSuchKey":
                logger.debug(f"Object {obj_name} already absent during cleanup")
            else:
                # Log only the exception type, not the message: the message may
                # contain MinIO credentials, endpoints, or other sensitive data.
                logger.warning(
                    "Failed to cleanup source object %s: %s",
                    obj_name,
                    type(e).__name__,
                )


@router.post("/upload", response_model=UploadResponse)
async def upload_scan(
    background_tasks: BackgroundTasks,
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
    dataset_subject_id: str | None = Form(default=None, max_length=64),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _rate_limit: None = Depends(enforce_upload_rate_limit),
):
    scan_id = str(uuid.uuid4())
    if isinstance(dataset_subject_id, str):
        dataset_subject_id = dataset_subject_id.strip() or None
    else:
        dataset_subject_id = None

    if (
        dataset_subject_id is not None
        and resolve_brats2020_cohort(dataset_subject_id) is None
    ):
        dataset_subject_id = None

        raise HTTPException(
            status_code=422,
            detail="dataset_subject_id must be a BraTS 2020 training subject ID",
        )
    bucket_name = "neuroscan-bucket"

    modality_files = {
        "t1": t1,
        "t1ce": t1ce,
        "t2": t2,
        "flair": flair,
    }

    # Track successfully uploaded objects for compensation
    uploaded_objects: List[str] = []

    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        os.makedirs("temp_uploads", exist_ok=True)

        modality_objects = {}
        for modality, file in modality_files.items():
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail=f"Modality {modality} must have a filename",
                )
            filename_lower = file.filename.lower()
            if filename_lower.endswith(".nii.gz"):
                file_type = ".nii.gz"
            elif filename_lower.endswith(".nii"):
                file_type = ".nii"
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Modality {modality} must be .nii or .nii.gz format"
                )

            # Validate NIfTI header before writing to disk
            if not _validate_nifti_header(file.file, file_type):
                raise HTTPException(
                    status_code=400,
                    detail=f"Modality {modality} is not a valid NIfTI file"
                )

            temp_path = os.path.join("temp_uploads", f"{scan_id}_{modality}{file_type}")

            # Write file with size limit enforcement
            total_bytes = 0
            try:
                with open(temp_path, "wb") as buffer:
                    total_bytes = _copyfileobj_with_limit(file.file, buffer, MAX_MODALITY_FILE_SIZE)
            except HTTPException:
                # Clean up partial file on size limit exceeded
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise

            # Validate minimum file size (sanity check - empty or tiny files are invalid)
            if total_bytes < 1024:  # Less than 1KB is suspicious for NIfTI
                os.remove(temp_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Modality {modality} file is too small to be a valid NIfTI volume"
                )

            object_name = f"{scan_id}/source_{modality}{file_type}"
            minio_client.fput_object(bucket_name, object_name, temp_path)
            os.remove(temp_path)

            # Track only after successful upload
            uploaded_objects.append(object_name)
            modality_objects[modality] = object_name

        # Create scan record with ownership
        scan_record = Scan(
            id=scan_id,
            filename=f"{scan_id}/source_study",  # Reference to study folder
            dataset_subject_id=dataset_subject_id,
            status="PENDING",
            user_id=current_user.firebase_uid,
        )
        db.add(scan_record)
        db.flush()  # Ensure Scan is persisted before ModalityFile FK references it

        # Create modality file records
        for modality, object_name in modality_objects.items():
            modality_record = ModalityFile(
                scan_id=scan_id,
                modality=modality,
                object_path=object_name,
            )
            db.add(modality_record)

        db.commit()

        # Pass modality object names to background task
        background_tasks.add_task(process_scan_task, scan_id, modality_objects)

        return UploadResponse(
            message="Four-modality study uploaded and dispatched successfully",
            scan_id=scan_id,
            status="PROCESSING",
        )

    except HTTPException:
        # Validation errors - clean up any uploaded objects
        if uploaded_objects:
            _cleanup_source_objects(bucket_name, uploaded_objects)
        raise
    except Exception as e:
        # Upload or database error - rollback DB first, then cleanup
        db.rollback()
        if uploaded_objects:
            _cleanup_source_objects(bucket_name, uploaded_objects)
        # Log diagnostics server-side (traceback) without returning exception
        # text, credentials, or internal details to the client.
        logger.exception("Unexpected error during scan upload")
        raise HTTPException(status_code=500, detail="Upload failed")


def get_scan_with_ownership(
    db: Session,
    scan_id: str,
    current_user: User,
) -> Scan:
    """
    Retrieve a scan by ID and verify the current user owns it.

    Centralizes the scan authorization check so every endpoint that reads
    or mutates a scan enforces ownership in one place. Fail closed: raises
    404 if the scan does not exist, 403 if the scan belongs to another user.

    Args:
        db: Database session
        scan_id: Scan identifier
        current_user: Authenticated user from Firebase

    Returns:
        Scan: The owned scan record

    Raises:
        HTTPException: 404 if scan not found, 403 if not owned by user
    """
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Enforce ownership
    if scan.user_id != current_user.firebase_uid:
        raise HTTPException(status_code=403, detail="Access denied: scan belongs to another user")

    return scan


@router.get("/status/{scan_id}", response_model=StatusResponse)
def get_scan_status(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan = get_scan_with_ownership(db, scan_id, current_user)

    return StatusResponse(
        scan_id=scan.id,
        status=scan.status,
    )

@router.get("/results/{scan_id}", response_model=ResultsResponse)
def get_scan_results(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan = get_scan_with_ownership(db, scan_id, current_user)
    cohort_metadata = resolve_brats2020_cohort(
        getattr(scan, "dataset_subject_id", None)
    )

    prediction = (
        db.query(Prediction)
        .filter(Prediction.scan_id == scan_id)
        .first()
    )

    return ResultsResponse(
        scan_id=scan.id,
        mask_url=get_presigned_url(scan.mask_path) if scan.mask_path else None,
        xai_url=get_presigned_url(scan.xai_path) if scan.xai_path else None,
        report_url=get_presigned_url(scan.report_path) if scan.report_path else None,
        tumor_detected=prediction.tumor_detected if prediction else False,
        max_tumor_probability=(
            prediction.max_tumor_probability if prediction else 0.0
        ),
        predicted_mask_pixels=(
            prediction.predicted_mask_pixels if prediction else None
        ),
        slice_coverage_percent=(
            prediction.slice_coverage_percent if prediction else None
        ),
        model_version=prediction.model_version_id if prediction else None,
        tumor_type=None,
        tumor_type_source="Not determined by ARMT-GAN segmentation",
        tumor_grade=None,
        tumor_grade_source=(
            "Not determined; no validated grading output is available"
        ),
        dataset_cohort_label=(
            cohort_metadata["cohort_label"] if cohort_metadata else None
        ),
        dataset_cohort_source=(
            "BraTS2020_TrainingData/name_mapping.csv; supplied subject ID; "
            "dataset metadata only, not an ARMT-GAN prediction or WHO grade"
            if cohort_metadata
            else None
        ),
    )


# =========================================================================
# Kaggle Classification Endpoints
# =========================================================================

# Kaggle-specific constants
KAGGLE_BUCKET = "neuroscan-bucket"
KAGGLE_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB for JPG/PNG
KAGGLE_ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
KAGGLE_CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")


def _validate_kaggle_image(file_obj: BinaryIO, file_ext: str) -> bool:
    """
    Validate a JPG/PNG image file for Kaggle upload.

    Basic validation: file extension, size, and PIL-readable.
    """
    original_pos = file_obj.tell()
    try:
        file_obj.seek(0, os.SEEK_END)
        total_bytes = file_obj.tell()
        file_obj.seek(0)
        if total_bytes > KAGGLE_MAX_FILE_SIZE:
            return False
        if total_bytes < 1024:  # Minimum 1KB
            return False

        # Try to open with PIL to verify it's a valid image
        from PIL import Image, UnidentifiedImageError
        file_obj.seek(0)
        try:
            img = Image.open(file_obj)
            img.verify()  # Verify it's a valid image
            file_obj.seek(0)
            return True
        except (UnidentifiedImageError, OSError):
            return False
    finally:
        file_obj.seek(original_pos)


def get_kaggle_scan_with_ownership(
    db: Session,
    scan_id: str,
    current_user: User,
) -> KaggleScan:
    """
    Retrieve a Kaggle scan by ID and verify the current user owns it.

    Fail closed: raises 404 if scan not found, 403 if scan belongs to another user.
    """
    scan = db.query(KaggleScan).filter(KaggleScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Enforce ownership
    if scan.user_id != current_user.firebase_uid:
        raise HTTPException(status_code=403, detail="Access denied: scan belongs to another user")

    return scan


@router.post("/kaggle/upload", response_model=KaggleUploadResponse)
async def upload_kaggle_scan(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _rate_limit: None = Depends(enforce_upload_rate_limit),
):
    """
    Upload a single JPG/PNG image for Kaggle brain tumor classification.

    Accepts .jpg, .jpeg, .png files up to 20MB.
    """
    scan_id = str(uuid.uuid4())
    bucket_name = KAGGLE_BUCKET

    # Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in KAGGLE_ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"File must be .jpg, .jpeg, or .png format"
        )

    # Validate image content
    if not _validate_kaggle_image(file.file, ""):
        raise HTTPException(
            status_code=400,
            detail="File is not a valid JPG/PNG image or exceeds size limits"
        )

    uploaded_objects = []
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        os.makedirs("temp_uploads", exist_ok=True)

        # Save to temp file
        file_ext = os.path.splitext(filename_lower)[1]
        temp_path = os.path.join("temp_uploads", f"{scan_id}_source{file_ext}")

        total_bytes = 0
        try:
            with open(temp_path, "wb") as buffer:
                total_bytes = _copyfileobj_with_limit(file.file, buffer, KAGGLE_MAX_FILE_SIZE)
        except HTTPException:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        object_name = f"kaggle/{scan_id}/source{file_ext}"
        minio_client.fput_object(bucket_name, object_name, temp_path)
        os.remove(temp_path)
        uploaded_objects.append(object_name)

        # Create Kaggle scan record
        scan_record = KaggleScan(
            id=scan_id,
            filename=f"kaggle/{scan_id}/source{file_ext}",
            status="PENDING",
            image_path=object_name,
            user_id=current_user.firebase_uid,
        )
        db.add(scan_record)
        db.commit()

        # Dispatch background classification task
        background_tasks.add_task(process_kaggle_task, scan_id, object_name)

        return KaggleUploadResponse(
            message="Kaggle image uploaded and dispatched successfully",
            scan_id=scan_id,
            status="PROCESSING",
        )

    except HTTPException:
        if uploaded_objects:
            _cleanup_source_objects(bucket_name, uploaded_objects)
        raise
    except Exception as e:
        db.rollback()
        if uploaded_objects:
            _cleanup_source_objects(bucket_name, uploaded_objects)
        logger.exception("Unexpected error during Kaggle scan upload")
        raise HTTPException(status_code=500, detail="Upload failed")


@router.get("/kaggle/status/{scan_id}", response_model=KaggleStatusResponse)
def get_kaggle_scan_status(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan = get_kaggle_scan_with_ownership(db, scan_id, current_user)
    return KaggleStatusResponse(scan_id=scan.id, status=scan.status)


@router.get("/kaggle/results/{scan_id}", response_model=KaggleResultsResponse)
def get_kaggle_scan_results(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan = get_kaggle_scan_with_ownership(db, scan_id, current_user)
    prediction = db.query(KagglePrediction).filter(KagglePrediction.scan_id == scan_id).first()

    return KaggleResultsResponse(
        scan_id=scan.id,
        class_index=prediction.class_index if prediction else None,
        class_name=prediction.class_name if prediction else None,
        probabilities=prediction.probabilities if prediction else None,
        model_version=prediction.model_version_id if prediction else None,
        image_url=get_presigned_url(scan.image_path) if scan.image_path else None,
    )