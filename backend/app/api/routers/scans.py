"""
Scans API Router.
Handles upload, status polling, and results retrieval for brain MRI scans.
All endpoint logic that was previously in main.py is consolidated here.
"""
import os
import uuid
import shutil
import logging
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


def _validate_nifti_header(file_obj: BinaryIO) -> bool:
    """
    Validate NIfTI file format by checking magic bytes.
    Handles both uncompressed (.nii) and gzipped (.nii.gz) formats.
    Reads minimal bytes without loading entire file.
    """
    # Save current position
    original_pos = file_obj.tell()
    try:
        # Read first 4 bytes for NIfTI-1/2 magic, or first 2 for gzip
        header = file_obj.read(4)
        file_obj.seek(0)
        
        if len(header) < 4:
            return False
        
        # Check for gzip magic (gzipped NIfTI)
        if header[:2] == GZIP_MAGIC:
            # For gzip, we'd need to decompress to check NIfTI magic
            # For now, accept gzip files with .nii.gz extension
            return True
        
        # Check NIfTI-1 magic (ni1) or NIfTI-2 magic (n+1)
        if header[:3] == NIFTI1_MAGIC or header[:3] == NIFTI2_MAGIC:
            return True
        
        return False
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
            if not _validate_nifti_header(file.file):
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
