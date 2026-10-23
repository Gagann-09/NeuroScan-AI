"""
Strict Pydantic response schemas for the Scans API.
These enforce a typed contract on all API responses.
"""
from pydantic import BaseModel, Field
from typing import Optional


class UploadRequest(BaseModel):
    """Request schema for POST /api/v1/scans/upload - four modalities required."""
    t1: str = Field(..., description="T1 modality file object name in storage")
    t1ce: str = Field(..., description="T1ce modality file object name in storage")
    t2: str = Field(..., description="T2 modality file object name in storage")
    flair: str = Field(..., description="FLAIR modality file object name in storage")


class UploadResponse(BaseModel):
    """Response schema for POST /api/v1/scans/upload."""
    message: str = Field(..., description="Human-readable upload status message")
    scan_id: str = Field(..., description="Unique identifier for the uploaded scan")
    status: str = Field(..., description="Current processing status")


class StatusResponse(BaseModel):
    """Response schema for GET /api/v1/scans/status/{scan_id}."""
    scan_id: str = Field(..., description="Unique identifier for the scan")
    status: str = Field(..., description="Current processing status")

class ResultsResponse(BaseModel):
    """Response schema for GET /api/v1/scans/results/{scan_id}."""
    scan_id: str = Field(..., description="Unique identifier for the scan")
    mask_url: Optional[str] = Field(
        None, description="Presigned URL for the segmentation mask overlay"
    )
    xai_url: Optional[str] = Field(
        None, description="Presigned URL for the input-gradient saliency map"
    )
    report_url: Optional[str] = Field(
        None, description="Presigned URL for the PDF report"
    )
    tumor_detected: bool = Field(
        False, description="Whether the segmentation pipeline detected tumor pixels"
    )
    max_tumor_probability: float = Field(
        0.0, description="Maximum tumor probability from model output (raw sigmoid)"
    )
    predicted_mask_pixels: Optional[int] = Field(
        None,
        description="Number of pixels at probability >= 0.5 in the selected 2D slice",
    )
    slice_coverage_percent: Optional[float] = Field(
        None,
        description="Percentage of selected 2D slice pixels at probability >= 0.5",
    )
    model_version: Optional[str] = Field(
        None, description="Model version identifier"
    )

    # ARMT-GAN produces a binary segmentation mask, not a tumor classification.
    tumor_type: Optional[str] = Field(
        None, description="Tumor type; null when not determined by a validated classifier"
    )
    tumor_type_source: str = Field(
        "Not determined by ARMT-GAN segmentation",
        description="Source or limitation for the tumor-type field"
    )
    tumor_grade: Optional[str] = Field(
        None, description="WHO grade; null when not determined by a validated grading model"
    )
    tumor_grade_source: str = Field(
        "Not determined; no validated grading output is available",
        description="Source or limitation for the tumor-grade field"
    )
    dataset_cohort_label: Optional[str] = Field(
        None,
        description="BraTS cohort label from the supplied subject ID; not a model prediction or WHO grade",
    )
    dataset_cohort_source: Optional[str] = Field(
        None,
        description="Provenance for dataset cohort metadata, when a mapping match exists",
    )

# --- Kaggle Classification Schemas ---

class KaggleUploadResponse(BaseModel):
    """Response schema for POST /api/v1/scans/kaggle/upload."""
    message: str = Field(..., description="Human-readable upload status message")
    scan_id: str = Field(..., description="Unique identifier for the uploaded scan")
    status: str = Field(..., description="Current processing status")


class KaggleStatusResponse(BaseModel):
    """Response schema for GET /api/v1/scans/kaggle/status/{scan_id}."""
    scan_id: str = Field(..., description="Unique identifier for the scan")
    status: str = Field(..., description="Current processing status")


class KaggleResultsResponse(BaseModel):
    """Response schema for GET /api/v1/scans/kaggle/results/{scan_id}."""
    scan_id: str = Field(..., description="Unique identifier for the scan")
    class_index: Optional[int] = Field(None, description="Predicted class index (0=glioma, 1=meningioma, 2=notumor, 3=pituitary)")
    class_name: Optional[str] = Field(None, description="Predicted class name")
    probabilities: Optional[dict] = Field(None, description="Class probabilities: glioma, meningioma, notumor, pituitary")
    model_version: Optional[str] = Field(None, description="Model version identifier")
    image_url: Optional[str] = Field(None, description="Presigned URL for the source image")
