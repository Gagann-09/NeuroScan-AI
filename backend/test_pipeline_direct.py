import time
from app.db.database import SessionLocal
from app.db.models import MRIScan
from app.services.ai_tasks import process_mri_scan

def run_direct():
    db = SessionLocal()
    pending = db.query(MRIScan).filter(MRIScan.status == "PENDING").first()
    
    if not pending:
        print("[Direct Test] No PENDING scans found in DB.")
        db.close()
        return

    print(f"[Direct Test] Starting direct execution for Scan ID: {pending.id} ({pending.file_type})")
    start = time.time()
    
    # Direct function execution (bypassing Redis queue)
    result = process_mri_scan(str(pending.id))
    
    elapsed = time.time() - start
    print(f"[Direct Test] Completed in {elapsed:.2f} seconds.")
    print(f"[Direct Test] Output Result: {result}")
    db.close()

if __name__ == "__main__":
    run_direct()