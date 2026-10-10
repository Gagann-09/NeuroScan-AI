"""
Scans API Router.
Handles upload, status polling, and results retrieval for brain MRI scans.
All endpoint logic that was previously in main.py is consolidated here.
"""
import os
import uuid
import shutil
import logging
import tempfile
import gzip
import zlib
from io import BytesIO
from typing import List, BinaryIO

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Form
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_current_user
from app.db.models import Scan, ModalityFile, Prediction, User
from app.core.storage import minio_client, get_presigned_url
from app.services.ai_tasks import process_scan_task
from app.schemas.scan_schema import UploadRequest, UploadResponse, StatusResponse, ResultsResponse

router = APIRouter(prefix="/api/v1/scans", tags=["scans"])

logger = logging.getLogger(__name__)

# Maximum file size per modality (bytes). BraTS volumes are ~36MB uncompressed,
# ~5-15MB compressed. Set generous limit to prevent abuse while allowing valid scans.
MAX_MODALITY_FILE_SIZE = 100 * 1024 * 1024  # 100 MB per modality
ALLOWED_EXTENSIONS = (".nii", ".nii.gz")

# NIfTI magic numbers for format validation
NIFTI1_MAGIC = b"ni1"
NIFTI2_MAGIC = b"n+1"
GZIP_MAGIC = b"\x1f\x8b"

# Maximum decompressed size to validate header (prevent decompression bombs)
# NIfTI header is at most 352 bytes (348 bytes header + 4 bytes magic)
# But we read a bit more to be safe for nibabel parsing
MAX_DECOMPRESSED_HEADER_BYTES = 4096  # 4KB - enough for full header + some data


def _validate_nifti_header(file_obj: BinaryIO, file_ext: str) -> bool:
    """
    Validate NIfTI file format by checking magic bytes and header structure.
    Handles both uncompressed (.nii) and gzipped (.nii.gz) formats.
    Reads minimal bytes without loading entire file.
    Performs proper gzip decompression and NIfTI header validation.
    """
    # Save current position
    original_pos = file_obj.tell()
    try:
        # Read first 4 bytes for initial magic check
        header = file_obj.read(4)
        file_obj.seek(0)
        
        if len(header) < 4:
            return False
        
        # Check for gzip magic (gzipped NIfTI)
        if header[:2] == GZIP_MAGIC:
            # For gzip files, we need to decompress and validate the NIfTI content
            return _validate_gzipped_nifti(file_obj)
        
        # Check NIfTI-1 magic (ni1) or NIfTI-2 magic (n+1)
        if header[:3] == NIFTI1_MAGIC or header[:3] == NIFTI2_MAGIC:
            return _validate_uncompressed_nifti(file_obj, header)
        
        return False
    except Exception:
        return False
    finally:
        file_obj.seek(original_pos)


def _validate_uncompressed_nifti(file_obj: BinaryIO, initial_header: bytes) -> bool:
    """
    Validate uncompressed NIfTI-1 or NIfTI-2 file by checking header structure.
    Uses nibabel for proper validation.
    """
    file_obj.seek(0)
    try:
        # Read enough bytes for nibabel to validate the header
        # NIfTI header is 352 bytes (348 header + 4 magic)
        header_data = file_obj.read(352)
        if len(header_data) < 352:
            return False
        
        # Write to temporary file for nibabel validation
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp.write(header_data)
            tmp_path = tmp.name
        
        try:
            # Try to load with nibabel - this validates header structure
            import nibabel as nib
            img = nib.load(tmp_path)
            # Check it's a valid NIfTI image
            if img.header is None:
                return False
            # Validate it's a NIfTI-1 or NIfTI-2 image
            if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
                return False
            # Basic sanity check on dimensions
            shape = img.shape
            if len(shape) < 3 or any(d <= 0 for d in shape[:3]):
                return False
            return True
        except Exception:
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception:
        return False


def _validate_gzipped_nifti(file_obj: BinaryIO) -> bool:
    """
    Validate gzipped NIfTI file by decompressing and checking NIfTI content.
    Bounded decompression to prevent decompression bombs.
    """
    file_obj.seek(0)
    try:
        # Read gzip data with size limit
        # We'll decompress in a bounded way to check the NIfTI header
        import gzip
        import tempfile
        
        # Read gzip data with size limit (bounded by MAX_MODALITY_FILE_SIZE)
        gzipped_data = file_obj.read(MAX_MODALITY_FILE_SIZE + 1)
        if len(gzipped_data) > MAX_MODALITY_FILE_SIZE:
            return False
        
        # Check if it's actually a valid gzip file by trying to decompress
        # We'll decompress just enough to validate the NIfTI header (max 4KB decompressed)
        try:
            with gzip.GzipFile(fileobj=BytesIO(gzipped_data), mode='rb') as gz:
                # Read bounded amount of decompressed data for header validation
                decompressed_header = gz.read(MAX_DECOMPRESSED_HEADER_BYTES)
                if len(decompressed_header) < 352:
                    return False
        except (gzip.BadGzipFile, OSError, EOFError, zlib.error):
            return False
        
        # Now validate the decompressed header as NIfTI
        import nibabel as nib
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
            tmp.write(decompressed_header)
            tmp_path = tmp.name
        
        try:
            img = nib.load(tmp_path)
            if img.header is None:
                return False
            if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
                return False
            shape = img.shape
            if len(shape) < 3 or any(d <= 0 for d in shape[:3]):
                return False
            return True
        except Exception:
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception:
        return False
    finally:
        file_obj.seek(0)


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
                logger.warning(f"Failed to cleanup source object {obj_name}: {e}")


@router.post("/upload", response_model=UploadResponse)
async def upload_scan(
    background_tasks: BackgroundTasks,
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan_id = str(uuid.uuid4())
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{scan_id}", response_model=StatusResponse)
def get_scan_status(
    scan_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    # Enforce ownership
    if scan.user_id != current_user.firebase_uid:
        raise HTTPException(status_code=403, detail="Access denied: scan belongs to another user")
    
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
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Enforce ownership
    if scan.user_id != current_user.firebase_uid:
        raise HTTPException(status_code=403, detail="Access denied: scan belongs to another user")

    prediction = db.query(Prediction).filter(Prediction.scan_id == scan_id).first()

    return ResultsResponse(
        scan_id=scan.id,
        mask_url=get_presigned_url(scan.mask_path) if scan.mask_path else None,
        xai_url=get_presigned_url(scan.xai_path) if scan.xai_path else None,
        report_url=get_presigned_url(scan.report_path) if scan.report_path else None,
        tumor_detected=prediction.tumor_detected if prediction else False,
        max_tumor_probability=prediction.max_tumor_probability if prediction else 0.0,
        model_version=prediction.model_version_id if prediction else None,
    )
