from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import shutil
import uuid
import os

from app.db.database import engine, Base, get_db
from app.db.models import MRIScan, Patient, User
from app.core.storage import minio_client
from app.core.celery_app import celery_app

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="NeuroScan AI API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "NeuroScan AI Backend is operational"}

@app.post("/api/v1/scans/upload")
async def upload_scan(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # 1. Determine file type
        ext = file.filename.split(".")[-1].lower()
        file_type = "NIFTI" if ext in ["nii", "gz"] else "IMAGE"

        # 2. Generate identifiers & paths
        scan_id = str(uuid.uuid4())
        patient_uuid = str(uuid.uuid4())[:8]
        bucket_name = "neuroscan-bucket"

        # Ensure MinIO bucket exists
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Save temporary file locally before uploading to MinIO
        os.makedirs("temp_uploads", exist_ok=True)
        temp_path = os.path.join("temp_uploads", f"{scan_id}_{file.filename}")
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Upload to MinIO
        object_name = f"PT-{patient_uuid}/{scan_id}/{file.filename}"
        minio_client.fput_object(bucket_name, object_name, temp_path)

        # Cleanup local temp file
        os.remove(temp_path)

        # 3. Ensure a default system user exists to satisfy foreign key constraints
        default_user = db.query(User).first()
        if not default_user:
            default_user = User(
                id=str(uuid.uuid4()),
                email="system@neuroscan.ai",
                hashed_password="hashed_dummy_password"
            )
            db.add(default_user)
            db.commit()
            db.refresh(default_user)

        # 4. Create patient record
        patient = Patient(
            id=str(uuid.uuid4()),
            patient_identifier=f"PT-{patient_uuid}"
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

        # 5. Create MRI Scan record in DB linked to the valid system user
        scan_record = MRIScan(
            id=scan_id,
            patient_id=patient.id,
            uploaded_by=default_user.id,  # Satisfies foreign key constraint to users table
            file_path=f"{bucket_name}/{object_name}",
            file_type=file_type,
            status="UPLOADED"
        )
        db.add(scan_record)
        db.commit()

        # 6. Dispatch to Celery background worker
        celery_app.send_task("process_mri_scan", args=[scan_id])

        return {
            "message": "File uploaded and dispatched successfully",
            "scan_id": scan_id,
            "status": "UPLOADED"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/scans/status/{scan_id}")
def get_scan_status(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(MRIScan).filter(MRIScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {
        "scan_id": scan.id,
        "status": scan.status,
        "file_type": scan.file_type
    }

@app.get("/api/v1/scans/results/{scan_id}")
def get_scan_results(scan_id: str, db: Session = Depends(get_db)):
    """Generates secure pre-signed URLs for the AI mask, XAI heatmap, and PDF report."""
    scan = db.query(MRIScan).filter(MRIScan.id == scan_id).first()
    if not scan or scan.status != "SEGMENTED":
        raise HTTPException(status_code=404, detail="Scan results not ready or not found")
    
    try:
        bucket_name, object_name = scan.file_path.split("/", 1)
        folder_prefix = object_name.rsplit('/', 1)[0]
        
        # Generate URLs valid for 2 hours
        mask_url = minio_client.presigned_get_object(bucket_name, f"{folder_prefix}/mask_{scan_id}.png", expires=timedelta(hours=2))
        xai_url = minio_client.presigned_get_object(bucket_name, f"{folder_prefix}/xai_{scan_id}.png", expires=timedelta(hours=2))
        report_url = minio_client.presigned_get_object(bucket_name, f"{folder_prefix}/report_{scan_id}.pdf", expires=timedelta(hours=2))
        
        return {
            "scan_id": scan_id,
            "mask_url": mask_url,
            "xai_url": xai_url,
            "report_url": report_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate asset URLs: {str(e)}")