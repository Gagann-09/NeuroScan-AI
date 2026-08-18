"""
Strict Pydantic response schemas for the Scans API.
These enforce a typed contract on all API responses.
"""
from pydantic import BaseModel, Field
from typing import Optional


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
    mask_url: Optional[str] = Field(None, description="Presigned URL for the segmentation mask overlay")
    xai_url: Optional[str] = Field(None, description="Presigned URL for the XAI activation map")
    report_url: Optional[str] = Field(None, description="Presigned URL for the clinical PDF report")
    tumor_detected: bool = Field(False, description="Whether a tumor was detected")
    anomaly_area_cm2: float = Field(0.0, description="Estimated anomaly area in cm²")
    confidence_score: float = Field(0.0, description="Model confidence score")
    who_grade: str = Field("N/A", description="WHO tumor grade classification")
