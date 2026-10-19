from sqlalchemy import Column, String, Boolean, Float, DateTime, Integer, ForeignKey, CheckConstraint
import sqlalchemy as sa
from datetime import datetime

# Import Base from base.py (declarative base without DB connection)
from app.db.base import Base


class User(Base):
    """User identity keyed by Firebase UID.
    
    Created lazily on first authenticated request. No password or auth credentials stored.
    """
    __tablename__ = "users"
    
    firebase_uid = Column(String, primary_key=True, index=True)
    email = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=True)


class Scan(Base):
    __tablename__ = "scans"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    status = Column(String, default="PENDING")
    mask_path = Column(String, nullable=True)
    xai_path = Column(String, nullable=True)
    report_path = Column(String, nullable=True)
    xai_raw_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Track when background processing actually started to detect stale/abandoned attempts
    processing_started_at = Column(DateTime, nullable=True)
    # Number of times the stale-PENDING recovery loop has re-dispatched this
    # scan. Incremented durably before each re-dispatch; capped at 3, after
    # which the scan is marked FAILED and not retried again by the loop.
    attempt_count = Column(Integer, nullable=False, server_default="0", default=0)
    # Owner of this scan (Firebase UID)
    user_id = Column(String, ForeignKey("users.firebase_uid"), index=True, nullable=True)


class ModalityFile(Base):
    __tablename__ = "modality_files"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(String, ForeignKey("scans.id"), index=True)
    modality = Column(String, index=True)  # t1, t1ce, t2, flair
    object_path = Column(String)

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(String, primary_key=True, index=True)
    checkpoint_path = Column(String, nullable=False)
    config_hash = Column(String, nullable=True)
    preprocessing_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(String, ForeignKey("scans.id"), index=True, nullable=False)
    model_version_id = Column(String, ForeignKey("model_versions.id"), index=True, nullable=True)
    tumor_detected = Column(Boolean, default=False)
    max_tumor_probability = Column(Float, nullable=True)
    dice = Column(Float, nullable=True)
    iou = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Artifact(Base):
    __tablename__ = "artifacts"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), index=True, nullable=False)
    type = Column(String, index=True, nullable=False)  # mask, xai, xai_raw, report, etc.
    object_path = Column(String, nullable=False)
    # Artifact lifecycle state: PENDING (created, upload not yet complete), COMPLETE (uploaded), FAILED (upload failed)
    status = Column(String, nullable=False, default="PENDING")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('PENDING', 'COMPLETE', 'FAILED')",
            name='ck_artifact_status_valid'
        ),
    )


class KaggleScan(Base):
    """Kaggle single-image classification scan.

    Distinct from BraTS Scan (4-modality NIfTI segmentation).
    """
    __tablename__ = "kaggle_scans"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    status = Column(String, default="PENDING")
    image_path = Column(String, nullable=True)  # MinIO object path for source JPG/PNG
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_started_at = Column(DateTime, nullable=True)
    user_id = Column(String, ForeignKey("users.firebase_uid"), index=True, nullable=True)


class KaggleModelVersion(Base):
    """Kaggle classifier model version provenance.

    Separate from BraTS ModelVersion to maintain clear dataset-provenance boundary.
    """
    __tablename__ = "kaggle_model_versions"

    id = Column(String, primary_key=True, index=True)
    checkpoint_path = Column(String, nullable=False)
    config_hash = Column(String, nullable=True)
    preprocessing_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class KagglePrediction(Base):
    """Kaggle classification prediction result.

    Contains class probabilities for the 4 tumor types: glioma, meningioma, notumor, pituitary.
    """
    __tablename__ = "kaggle_predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(String, ForeignKey("kaggle_scans.id"), index=True, nullable=False)
    model_version_id = Column(String, ForeignKey("kaggle_model_versions.id"), index=True, nullable=True)
    class_index = Column(Integer, nullable=True)  # 0=glioma, 1=meningioma, 2=notumor, 3=pituitary
    class_name = Column(String, nullable=True)
    probabilities = Column(sa.JSON, nullable=True)  # {"glioma": 0.1, "meningioma": 0.2, "notumor": 0.6, "pituitary": 0.1}
    created_at = Column(DateTime, default=datetime.utcnow)
