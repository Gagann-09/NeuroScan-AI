# backend/ingest_datasets.py
import os
from pathlib import Path
from app.db.database import SessionLocal
from app.db.models import Patient, MRIScan, User
from app.core.storage import minio_client

# Resolve the path to your datasets folder
DATASETS_DIR = Path("../datasets").resolve()

def ingest_local_data():
    db = SessionLocal()

    admin_user = db.query(User).filter(User.email == "admin@neuroscan.ai").first()
    if not admin_user:
        print("[Error] Admin user not found. Run init_system.py first.")
        return

    if not DATASETS_DIR.exists():
        print(f"[Error] Datasets directory not found at: {DATASETS_DIR}")
        return

    print(f"--- Initiating Secure Ingestion from {DATASETS_DIR} ---")

    # Recursively scan all files
    for filepath in DATASETS_DIR.rglob("*"):
        if not filepath.is_file():
            continue

        path_parts = filepath.parts
        file_name = filepath.name
        
        # Safely extract extension (handling .nii.gz)
        ext = "".join(filepath.suffixes).lower() if ".nii.gz" in str(filepath).lower() else filepath.suffix.lower()

        # Route 1: BraTS Data (3D NIfTI)
        if "BraTS_data" in path_parts and ext in [".nii", ".nii.gz"]:
            bucket_name = "brats-scans"
            file_type = "NIFTI"
            # Parent folder is the patient ID (e.g., BraTS20_Training_001)
            patient_identifier = f"PT-{filepath.parent.name.upper()}"

        # Route 2: Kaggle Data (2D Images)
        elif "Kaggle_data" in path_parts and ext in [".jpg", ".jpeg", ".png"]:
            bucket_name = "kaggle-scans"
            file_type = "IMAGE"
            # Since Kaggle slices don't have patient folders, use the filename as the ID (e.g., PT-TE-AUG-ME_1)
            patient_identifier = f"PT-{filepath.stem.upper()}"
            
        else:
            continue # Skip unknown files

        # 1. Register Patient in Database
        patient = db.query(Patient).filter(Patient.patient_identifier == patient_identifier).first()
        if not patient:
            patient = Patient(patient_identifier=patient_identifier)
            db.add(patient)
            db.commit()
            db.refresh(patient)

        # 2. Upload to MinIO Object Storage
        minio_object_name = f"{patient_identifier}/{file_name}"
        full_storage_uri = f"{bucket_name}/{minio_object_name}"

        # Prevent duplicate uploads if script is run twice
        existing_scan = db.query(MRIScan).filter(MRIScan.file_path == full_storage_uri).first()
        if existing_scan:
            continue

        print(f"[Uploading] {file_name} -> {bucket_name} (Patient: {patient_identifier})")
        try:
            minio_client.fput_object(
                bucket_name,
                minio_object_name,
                str(filepath)
            )

            # 3. Register Scan in Database
            new_scan = MRIScan(
                patient_id=patient.id,
                uploaded_by=admin_user.id,
                file_path=full_storage_uri,
                file_type=file_type,
                status="PENDING" 
            )
            db.add(new_scan)
            db.commit()
        except Exception as e:
            print(f"[Failed] {file_name}: {str(e)}")
            db.rollback()

    print("--- Ingestion Pipeline Complete ---")
    db.close()

if __name__ == "__main__":
    ingest_local_data()