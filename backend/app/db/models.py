import uuid
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False) # Admin, Radiologist, System
    mfa_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_identifier = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class MRIScan(Base):
    __tablename__ = "mri_scans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    file_path = Column(String(512), nullable=False) # MinIO/S3 URI
    file_type = Column(String(10), nullable=False) # DICOM, NIFTI, JPG
    status = Column(String(50), default='PENDING') # PENDING, PROCESSING, COMPLETED, FAILED
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scan_id = Column(UUID(as_uuid=True), ForeignKey("mri_scans.id"), nullable=False)
    tumor_detected = Column(Boolean, nullable=False)
    tumor_type = Column(String(100)) # Glioma, Meningioma, Pituitary, Normal
    confidence_score = Column(Numeric(5, 4))
    accuracy_score = Column(Numeric(5, 4))
    segmentation_mask_path = Column(String(512)) # MinIO/S3 URI
    highlighted_mri_path = Column(String(512))   # MinIO/S3 URI
    gradcam_path = Column(String(512))           # MinIO/S3 URI
    shap_path = Column(String(512))              # MinIO/S3 URI
    saliency_path = Column(String(512))          # MinIO/S3 URI
    created_at = Column(DateTime(timezone=True), server_default=func.now())