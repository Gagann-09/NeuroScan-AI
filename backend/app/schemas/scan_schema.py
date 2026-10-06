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
    mask_url: Optional[str] = Field(None, description="Presigned URL for the segmentation mask overlay")
    xai_url: Optional[str] = Field(None, description="Presigned URL for the XAI activation map")
    report_url: Optional[str] = Field(None, description="Presigned URL for the PDF report")
    tumor_detected: bool = Field(False, description="Whether a tumor was detected")
    max_tumor_probability: float = Field(0.0, description="Maximum tumor probability from model output (raw sigmoid)")
    model_version: Optional[str] = Field(None, description="Model version identifier")
