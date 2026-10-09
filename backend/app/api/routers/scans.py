"""
Scans API Router.
Handles upload, status polling, and results retrieval for brain MRI scans.
All endpoint logic that was previously in main.py is consolidated here.
"""
import os
import uuid
import shutil

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Form
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db.models import Scan, ModalityFile, Prediction
from app.core.storage import minio_client, get_presigned_url
from app.services.ai_tasks import process_scan_task
from app.schemas.scan_schema import UploadRequest, UploadResponse, StatusResponse, ResultsResponse

router = APIRouter(prefix="/api/v1/scans", tags=["scans"])


@router.post("/upload", response_model=UploadResponse)
async def upload_scan(
    background_tasks: BackgroundTasks,
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        scan_id = str(uuid.uuid4())
        bucket_name = "neuroscan-bucket"
        
        modality_files = {
            "t1": t1,
            "t1ce": t1ce,
            "t2": t2,
            "flair": flair,
        }
        
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
            
            temp_path = os.path.join("temp_uploads", f"{scan_id}_{modality}{file_type}")
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            object_name = f"{scan_id}/source_{modality}{file_type}"
            minio_client.fput_object(bucket_name, object_name, temp_path)
            os.remove(temp_path)
            
            modality_objects[modality] = object_name
        
        # Create scan record
        scan_record = Scan(
            id=scan_id,
            filename=f"{scan_id}/source_study",  # Reference to study folder
            status="PENDING",
        )
        db.add(scan_record)
        
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
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{scan_id}", response_model=StatusResponse)
def get_scan_status(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return StatusResponse(
        scan_id=scan.id,
        status=scan.status,
    )


@router.get("/results/{scan_id}", response_model=ResultsResponse)
def get_scan_results(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

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
