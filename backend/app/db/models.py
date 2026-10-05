from sqlalchemy import Column, String, Boolean, Float, DateTime, Integer, ForeignKey
from datetime import datetime

# CRUCIAL FIX: Import the exact Base used by the database connection
from app.db.database import Base

class Scan(Base):
    __tablename__ = "scans"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    status = Column(String, default="PENDING")
    mask_path = Column(String, nullable=True)
    xai_path = Column(String, nullable=True)
    report_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModalityFile(Base):
    __tablename__ = "modality_files"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(String, ForeignKey("scans.id"), index=True)
    modality = Column(String, index=True)  # t1, t1ce, t2, flair
    object_path = Column(String)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(String, index=True)
    tumor_detected = Column(Boolean, default=False)
    anomaly_area_cm2 = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    who_grade = Column(String, nullable=True)