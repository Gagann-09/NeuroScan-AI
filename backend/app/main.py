import os
import uuid
import shutil
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.db.database import engine, Base, get_db
from app.db.models import Scan, Prediction
from app.core.storage import minio_client, get_presigned_url
from app.services.ai_tasks import process_scan_task

Base.metadata.drop_all(bind=engine)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="NeuroScan AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/scans/upload")
async def upload_scan(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    try:
        scan_id = str(uuid.uuid4())
        bucket_name = "neuroscan-bucket"
        
        # FIXED: Accurately extract the exact file extension
        filename_lower = file.filename.lower()
        if filename_lower.endswith(".nii.gz"):
            file_type = ".nii.gz"
        elif filename_lower.endswith(".nii"):
            file_type = ".nii"
        elif filename_lower.endswith(".png"):
            file_type = ".png"
        else:
            file_type = ".jpg"

        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        os.makedirs("temp_uploads", exist_ok=True)
        temp_path = os.path.join("temp_uploads", f"{scan_id}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        object_name = f"{scan_id}/source_{file.filename}"
        minio_client.fput_object(bucket_name, object_name, temp_path)
        os.remove(temp_path)

        scan_record = Scan(
            id=scan_id,
            filename=object_name,
            status="PROCESSING"
        )
        db.add(scan_record)
        db.commit()

        background_tasks.add_task(process_scan_task, scan_id, object_name, file_type)

        return {
            "message": "File uploaded and dispatched successfully",
            "scan_id": scan_id,
            "status": "PROCESSING"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/scans/status/{scan_id}")
def get_scan_status(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {
        "scan_id": scan.id,
        "status": scan.status
    }

@app.get("/api/v1/scans/results/{scan_id}")
def get_scan_results(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    prediction = db.query(Prediction).filter(Prediction.scan_id == scan_id).first()
    
    metrics = {
        "tumor_detected": prediction.tumor_detected if prediction else False,
        "anomaly_area_cm2": prediction.anomaly_area_cm2 if prediction else 0.0,
        "confidence_score": prediction.confidence_score if prediction else 0.0,
        "who_grade": prediction.who_grade if prediction else "N/A"
    }

    return {
        "scan_id": scan.id,
        "mask_url": get_presigned_url(scan.mask_path) if scan.mask_path else None,
        "xai_url": get_presigned_url(scan.xai_path) if scan.xai_path else None,
        "report_url": get_presigned_url(scan.report_path) if scan.report_path else None,
        **metrics 
    }