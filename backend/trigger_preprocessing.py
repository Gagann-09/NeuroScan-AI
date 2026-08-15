# backend/trigger_preprocessing.py
from app.db.database import SessionLocal
from app.db.models import MRIScan
from app.services.ai_tasks import process_mri_scan

def trigger_batch():
    db = SessionLocal()
    
    # Fetch up to 5 PENDING scans to verify the pipeline
    pending_scans = db.query(MRIScan).filter(MRIScan.status == "PENDING").limit(5).all()

    if not pending_scans:
        print("[Trigger] No PENDING scans found. All data is processed.")
        db.close()
        return

    print(f"[Trigger] Dispatching {len(pending_scans)} scans to the AI Worker...")
    for scan in pending_scans:
        print(f" -> Queueing Scan ID: {scan.id} (Type: {scan.file_type})")
        # .delay() securely pushes the task to the Redis queue
        process_mri_scan.delay(str(scan.id))

    print("[Trigger] Dispatch complete. Watch your original Celery terminal for live PyTorch logs!")
    db.close()

if __name__ == "__main__":
    trigger_batch()