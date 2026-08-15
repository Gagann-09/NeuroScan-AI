from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import uuid
import os

from app.db.database import SessionLocal
from app.db.models import MRIScan, User, Patient # ADDED Patient IMPORT
from app.core.storage import minio_client
from app.services.ai_tasks import process_mri_scan

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload")
async def upload_scan(file: UploadFile = File(...), db: Session = Depends(get_db)):
    
    admin_user = db.query(User).filter(User.email == "admin@neuroscan.ai").first()
    if not admin_user:
        raise HTTPException(status_code=500, detail="System not initialized properly.")

    # FIX: Dynamically create or fetch a valid web patient to get a real UUID
    web_patient = db.query(Patient).filter(Patient.patient_identifier == "PT-WEB-DEFAULT").first()
    if not web_patient:
        web_patient = Patient(patient_identifier="PT-WEB-DEFAULT")
        db.add(web_patient)
        db.commit()
        db.refresh(web_patient)

    ext = file.filename.split(".")[-1].lower()
    if ext in ["nii", "gz"]:
        bucket_name = "brats-scans"
        file_type = "NIFTI"
    elif ext in ["jpg", "jpeg", "png"]:
        bucket_name = "kaggle-scans"
        file_type = "IMAGE"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    web_patient_id = f"PT-WEB-{str(uuid.uuid4())[:8].upper()}"
    minio_object_name = f"{web_patient_id}/{file.filename}"
    full_storage_uri = f"{bucket_name}/{minio_object_name}"

    try:
        minio_client.put_object(
            bucket_name,
            minio_object_name,
            file.file,
            length=-1,
            part_size=10*1024*1024
        )

        # Use the valid UUID from the web_patient
        new_scan = MRIScan(
            patient_id=web_patient.id, 
            uploaded_by=admin_user.id,
            file_path=full_storage_uri,
            file_type=file_type,
            status="PENDING"
        )
        db.add(new_scan)
        db.commit()
        db.refresh(new_scan)

        process_mri_scan.delay(str(new_scan.id))

        return {
            "message": "Scan uploaded and AI processing started.",
            "scan_id": str(new_scan.id),
            "status": "PROCESSING"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{scan_id}")
def get_scan_status(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(MRIScan).filter(MRIScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
        
    return {
        "scan_id": str(scan.id),
        "status": scan.status,
        "file_type": scan.file_type
    }