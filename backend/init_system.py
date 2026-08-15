from app.db.database import SessionLocal
from app.db.models import User
from app.core.storage import init_buckets
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def init_admin():
    db = SessionLocal()
    admin_email = "admin@neuroscan.ai"
    existing_admin = db.query(User).filter(User.email == admin_email).first()

    if not existing_admin:
        hashed_password = pwd_context.hash("Admin@123!")
        admin_user = User(
            email=admin_email,
            password_hash=hashed_password,
            role="Admin"
        )
        db.add(admin_user)
        db.commit()
        print(f"[Database] System Administrator '{admin_email}' created.")
    else:
        print("[Database] System Administrator already exists.")
    db.close()

if __name__ == "__main__":
    print("--- Initializing NeuroScan AI Infrastructure ---")
    init_buckets()
    init_admin()
    print("--- Initialization Complete ---")