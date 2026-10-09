import os
import cv2
import torch
import numpy as np
import tempfile
import hashlib
from PIL import Image
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal

from app.db.database import SessionLocal
from app.db.models import Scan, ModalityFile, Prediction, ModelVersion, Artifact
from app.core.storage import minio_client, upload_file_to_minio
from app.services.reporting import generate_segmentation_report
from ai_pipeline.models.armt_gan import ARMTGenerator2D
from app.services.xai import generate_gradient_saliency, compute_segmentation_alignment, XAIProvenance
from ai_pipeline.preprocessing import preprocess_brats_study, PreprocessingConfig

logger = logging.getLogger(__name__)


# ── LAZY MODEL LOADING ──
_DEVICE = None
_GLOBAL_MODEL = None
_WEIGHTS_PATH = None
_CHECKPOINT_IDENTIFIER = None
_MODEL_VERSION_ID = None
_DEFAULT_PREPROCESSING_CONFIG = None
_PREPROCESSING_VERSION = None
_MODEL_VERSION_CONFIG_HASH = "arch:armt-gan-2d-unet|generator:lightweight|discriminator:conditional-patchgan|loss:l1_100_adv_1"


def _get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def _get_weights_path():
    global _WEIGHTS_PATH
    if _WEIGHTS_PATH is None:
        _WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../../ai_pipeline/weights/generator_latest.pth")
    return _WEIGHTS_PATH


def _get_global_model():
    """Lazily load and return the global ARMT-GAN model."""
    global _GLOBAL_MODEL, _CHECKPOINT_IDENTIFIER
    if _GLOBAL_MODEL is None:
        logger.info("Initializing ARMT-GAN model into global memory...")
        _GLOBAL_MODEL = ARMTGenerator2D().to(_get_device())
        weights_path = _get_weights_path()
        if os.path.exists(weights_path):
            _GLOBAL_MODEL.load_state_dict(torch.load(weights_path, map_location=_get_device()))
        _GLOBAL_MODEL.eval()
        _CHECKPOINT_IDENTIFIER = os.path.basename(weights_path) if os.path.exists(weights_path) else "unknown"
        logger.info("ARMT-GAN model successfully cached in memory.")
    return _GLOBAL_MODEL


def _get_checkpoint_identifier():
    _get_global_model()  # Ensures model is loaded and identifier set
    return _CHECKPOINT_IDENTIFIER


def _get_model_version_id():
    global _MODEL_VERSION_ID
    if _MODEL_VERSION_ID is None:
        weights_path = _get_weights_path()
        _MODEL_VERSION_ID = _compute_model_version_id(weights_path)
    return _MODEL_VERSION_ID


def _get_default_preprocessing_config():
    global _DEFAULT_PREPROCESSING_CONFIG
    if _DEFAULT_PREPROCESSING_CONFIG is None:
        _DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig(image_size=224)
    return _DEFAULT_PREPROCESSING_CONFIG


def _get_preprocessing_version():
    global _PREPROCESSING_VERSION
    if _PREPROCESSING_VERSION is None:
        _PREPROCESSING_VERSION = _compute_preprocessing_version(_get_default_preprocessing_config())
    return _PREPROCESSING_VERSION


def _get_model_version_config_hash():
    return _MODEL_VERSION_CONFIG_HASH


def _compute_model_version_id(weights_path: str) -> str:
    """
    Compute a deterministic model version ID from the model weights file.
    Uses SHA256 hash of the weights file content.
    """
    import hashlib
    try:
        with open(weights_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return f"model-{file_hash[:16]}"
    except Exception:
        # Fallback if file doesn't exist or can't be read
        return "model-unknown"


def _compute_preprocessing_version(config: "PreprocessingConfig") -> str:
    """
    Compute a deterministic preprocessing version from the configuration.
    Uses SHA256 hash of the canonical config representation.
    """
    import hashlib
    import json

    # Create a canonical representation of the config
    config_dict = {
        "image_size": config.image_size,
        # Add other config fields as needed
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return f"preproc-{hashlib.sha256(config_str.encode()).hexdigest()[:16]}"


class ScanClaimResult:
    """Result of attempting to claim a scan for processing."""
    
    def __init__(
        self,
        success: bool,
        reason: Literal["CLAIMED", "ALREADY_PROCESSING", "ALREADY_COMPLETE", "NOT_FOUND"] | None = None,
        scan_id: str | None = None,
    ):
        self.success = success
        self.reason = reason
        self.scan_id = scan_id
    
    def __bool__(self) -> bool:
        return self.success
    
    def __repr__(self) -> str:
        if self.success:
            return f"ScanClaimResult(success=True, scan_id={self.scan_id})"
        return f"ScanClaimResult(success=False, reason={self.reason}, scan_id={self.scan_id})"


def claim_scan_for_processing(db: Session, scan_id: str) -> ScanClaimResult:
    """
    Atomically claim a scan for processing using PostgreSQL row-level locking.
    
    Uses SELECT FOR UPDATE NOWAIT to acquire an exclusive lock on the Scan row.
    If another transaction holds the lock, returns immediately with failure.
    
    State transitions:
    - PENDING    → PROCESSING (claimed, processing_started_at set)
    - FAILED     → PROCESSING (claimed, processing_started_at set)  
    - PROCESSING → REJECTED (another worker holds the lock)
    - SEGMENTED  → REJECTED (already complete)
    - NOT FOUND  → REJECTED (scan does not exist)
    
    Args:
        db: Database session
        scan_id: Scan identifier to claim
        
    Returns:
        ScanClaimResult with success status and reason
    """
    # Attempt to acquire row lock with NOWAIT - fails immediately if locked
    try:
        row = db.execute(
            text("SELECT id, status FROM scans WHERE id = :scan_id FOR UPDATE NOWAIT"),
            {"scan_id": scan_id}
        ).fetchone()
    except Exception as e:
        # Lock not available - another transaction holds it (PROCESSING)
        if "could not obtain lock" in str(e).lower() or "lock_not_available" in str(e).lower():
            return ScanClaimResult(success=False, reason="ALREADY_PROCESSING", scan_id=scan_id)
        raise
    
    if not row:
        return ScanClaimResult(success=False, reason="NOT_FOUND", scan_id=scan_id)
    
    current_status = row.status
    
    # Check current state and decide
    if current_status == "SEGMENTED":
        return ScanClaimResult(success=False, reason="ALREADY_COMPLETE", scan_id=scan_id)
    
    if current_status == "PROCESSING":
        # Another worker already claimed it (we got the lock but status is PROCESSING)
        # This shouldn't happen with NOWAIT but handle defensively
        return ScanClaimResult(success=False, reason="ALREADY_PROCESSING", scan_id=scan_id)
    
    # Claim the scan: PENDING or FAILED
    now = datetime.now(timezone.utc)
    db.execute(
        text("UPDATE scans SET status = 'PROCESSING', processing_started_at = :now WHERE id = :scan_id"),
        {"scan_id": scan_id, "now": now}
    )
    db.flush()
    
    return ScanClaimResult(success=True, reason="CLAIMED", scan_id=scan_id)


class ArtifactStateTransition:
    """Result of attempting to transition an artifact's lifecycle state."""
    
    def __init__(
        self,
        success: bool,
        reason: Literal[
            "TRANSITIONED",
            "ALREADY_COMPLETE",
            "ALREADY_FAILED",
            "INVALID_TRANSITION",
            "NOT_FOUND",
            "INVALID_STATE"
        ] | None = None,
        artifact_id: int | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
    ):
        self.success = success
        self.reason = reason
        self.artifact_id = artifact_id
        self.from_state = from_state
        self.to_state = to_state
    
    def __bool__(self) -> bool:
        return self.success
    
    def __repr__(self) -> str:
        if self.success:
            return f"ArtifactStateTransition(success=True, artifact_id={self.artifact_id}, {self.from_state} -> {self.to_state})"
        return f"ArtifactStateTransition(success=False, reason={self.reason}, artifact_id={self.artifact_id})"


# Valid lifecycle transitions
VALID_ARTIFACT_TRANSITIONS = {
    "PENDING": {"COMPLETE", "FAILED"},
    "COMPLETE": set(),  # Terminal state - no transitions allowed
    "FAILED": set(),    # Terminal state - no transitions allowed
}


def transition_artifact_state(
    db: Session,
    artifact_id: int,
    target_state: Literal["PENDING", "COMPLETE", "FAILED"],
) -> ArtifactStateTransition:
    """
    Atomically transition an artifact's lifecycle state using PostgreSQL row-level locking.
    
    Uses SELECT FOR UPDATE NOWAIT to acquire an exclusive lock on the Artifact row.
    If another transaction holds the lock, raises an exception.
    
    Valid transitions (per migration 005 CHECK constraint and design):
    - PENDING -> COMPLETE (successful upload/completion)
    - PENDING -> FAILED (failed upload/operation)
    
    Invalid transitions (rejected explicitly):
    - COMPLETE -> any state (terminal)
    - FAILED -> any state (terminal)
    - Any -> PENDING (no backward transitions)
    - Any -> unknown state
    
    Args:
        db: Database session
        artifact_id: Artifact identifier to transition
        target_state: Target lifecycle state
        
    Returns:
        ArtifactStateTransition with success status, reason, and state info
    """
    # Validate target state is a known lifecycle state
    if target_state not in VALID_ARTIFACT_TRANSITIONS:
        return ArtifactStateTransition(
            success=False,
            reason="INVALID_STATE",
            artifact_id=artifact_id,
            to_state=target_state,
        )
    
    # Attempt to acquire row lock with NOWAIT - fails immediately if locked
    try:
        row = db.execute(
            text("SELECT id, status FROM artifacts WHERE id = :artifact_id FOR UPDATE NOWAIT"),
            {"artifact_id": artifact_id}
        ).fetchone()
    except Exception as e:
        # Lock not available - another transaction holds it
        if "could not obtain lock" in str(e).lower() or "lock_not_available" in str(e).lower():
            return ArtifactStateTransition(
                success=False,
                reason="INVALID_TRANSITION",
                artifact_id=artifact_id,
                to_state=target_state,
            )
        raise
    
    if not row:
        return ArtifactStateTransition(
            success=False,
            reason="NOT_FOUND",
            artifact_id=artifact_id,
            to_state=target_state,
        )
    
    current_state = row.status
    
    # Check if transition is valid
    allowed_targets = VALID_ARTIFACT_TRANSITIONS.get(current_state, set())
    if target_state not in allowed_targets:
        return ArtifactStateTransition(
            success=False,
            reason="INVALID_TRANSITION",
            artifact_id=artifact_id,
            from_state=current_state,
            to_state=target_state,
        )
    
    # Perform the valid transition
    db.execute(
        text("UPDATE artifacts SET status = :target_state WHERE id = :artifact_id"),
        {"artifact_id": artifact_id, "target_state": target_state}
    )
    db.flush()
    
    return ArtifactStateTransition(
        success=True,
        reason="TRANSITIONED",
        artifact_id=artifact_id,
        from_state=current_state,
        to_state=target_state,
    )


def _get_or_create_model_version(db: Session, version_id: str, checkpoint_path: str, config_hash: str, preprocessing_version: str) -> ModelVersion:
    """
    Get existing ModelVersion or create new one.
    Ensures deterministic reuse of the same model version for the same checkpoint.
    """
    model_version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
    if model_version is None:
        model_version = ModelVersion(
            id=version_id,
            checkpoint_path=checkpoint_path,
            config_hash=config_hash,
            preprocessing_version=preprocessing_version,
        )
        db.add(model_version)
        db.flush()  # Ensure ID is available
        logger.info(f"Created new ModelVersion: {version_id}")
    return model_version


def generate_clinical_overlays(original_pil: Image.Image, mask_tensor: np.ndarray, xai_tensor: np.ndarray):
    """Generate visualization overlays for segmentation and XAI."""
    orig_np = np.array(original_pil.convert('RGB'))
    orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

    mask_np = mask_tensor.squeeze()
    
    if mask_np.max() > mask_np.min():
        norm_mask = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min())
    else:
        norm_mask = mask_np

    thresh_val = np.percentile(norm_mask, 85) if np.max(norm_mask) > 0.1 else 0.2
    _, binary_mask = cv2.threshold((norm_mask * 255).astype(np.uint8), int(thresh_val * 255), 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    if cv2.countNonZero(clean_mask) < 5:
        _, clean_mask = cv2.threshold((norm_mask * 255).astype(np.uint8), 50, 255, cv2.THRESH_BINARY)

    glow_mask = cv2.GaussianBlur(clean_mask, (21, 21), 0)
    glow_mask_float = glow_mask.astype(float) / 255.0

    color_layer = np.zeros_like(orig_bgr)
    color_layer[:] = [150, 255, 50] 

    alpha = 0.70
    mask_3d = np.repeat(glow_mask_float[:, :, np.newaxis], 3, axis=2)
    seg_overlay = (orig_bgr * (1 - mask_3d * alpha) + color_layer * (mask_3d * alpha)).astype(np.uint8)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_overlay, contours, -1, (180, 255, 100), 2)

    heatmap_np = xai_tensor.squeeze()
    if heatmap_np.max() > heatmap_np.min():
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min())

    context_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    localized_area = cv2.dilate(clean_mask, context_kernel, iterations=1)
    localized_area_float = localized_area.astype(float) / 255.0

    localized_heatmap = heatmap_np * localized_area_float
    if np.max(localized_heatmap) > 0:
        localized_heatmap = localized_heatmap / np.max(localized_heatmap)

    heatmap_colored = cv2.applyColorMap((localized_heatmap * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    xai_alpha_mask = localized_heatmap[:, :, np.newaxis]
    xai_overlay = (orig_bgr * (1 - xai_alpha_mask * 0.7) + heatmap_colored * (xai_alpha_mask * 0.7)).astype(np.uint8)

    seg_final_pil = Image.fromarray(cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB))
    xai_final_pil = Image.fromarray(cv2.cvtColor(xai_overlay, cv2.COLOR_BGR2RGB))
    
    tumor_area_px = cv2.countNonZero(clean_mask)
    # No heuristic cm² conversion - pixel count only for internal thresholding
    tumor_detected = tumor_area_px > 0
    
    return seg_final_pil, xai_final_pil, tumor_detected, norm_mask


def create_reference_image(modality_arrays: dict[str, np.ndarray], slice_index: int) -> Image.Image:
    """
    Create a reference RGB image from the 4 modalities for visualization.
    Uses FLAIR as base, overlays T1ce for contrast enhancement.
    """
    # Use FLAIR as the base (good for tumor visualization)
    flair_slice = modality_arrays["flair"][:, :, slice_index]
    
    # Normalize for display
    flair_display = flair_slice.copy()
    if flair_display.max() > flair_display.min():
        flair_display = (flair_display - flair_display.min()) / (flair_display.max() - flair_display.min())
    flair_display = (flair_display * 255).astype(np.uint8)
    
    # Create RGB from FLAIR (grayscale -> RGB)
    rgb = np.stack([flair_display, flair_display, flair_display], axis=2)
    
    return Image.fromarray(rgb, mode="RGB")


def process_scan_task(scan_id: str, modality_objects: dict[str, str]):
    """
    Process a 4-modality BraTS study through the ARMT-GAN pipeline.
    
    Args:
        scan_id: Unique scan identifier
        modality_objects: Dict mapping modality name to MinIO object path
            e.g., {"t1": "scan_id/source_t1.nii.gz", ...}
    """
    db: Session = SessionLocal()
    local_paths = {}
    try:
        logger.info(f"Starting 4-modality inference for Scan ID: {scan_id}")
        
        # Attempt to claim the scan for processing
        claim_result = claim_scan_for_processing(db, scan_id)
        if not claim_result.success:
            logger.info(f"Scan {scan_id} not claimed: {claim_result.reason}")
            return  # Exit early — another worker claimed it, or already complete/failed/not found

        # Download all 4 modalities from MinIO
        with tempfile.TemporaryDirectory() as tmpdir:
            for modality, object_name in modality_objects.items():
                local_path = os.path.join(tmpdir, f"{modality}.nii.gz")
                minio_client.fget_object("neuroscan-bucket", object_name, local_path)
                local_paths[modality] = local_path
            
            # Create a patient directory structure for preprocessing
            patient_dir = Path(tmpdir)
            
            # Rename files to match BraTS naming convention expected by preprocessor
            for modality, local_path in local_paths.items():
                expected_name = f"study_{modality}.nii.gz"
                expected_path = patient_dir / expected_name
                os.rename(local_path, expected_path)
                local_paths[modality] = expected_path
            
            # Use shared preprocessing pipeline (matches training exactly)
            config = PreprocessingConfig(image_size=224)
            image_tensor, mask_tensor, metadata = preprocess_brats_study(
                patient_dir=patient_dir,
                config=config,
            )
            
            # Move to device
            image_tensor = image_tensor.to(_get_device())
            
            # Load segmentation if available for reference image
            seg_path = None
            for p in patient_dir.iterdir():
                if "seg" in p.name.lower():
                    seg_path = p
                    break
            
            # Create reference image for visualization (from FLAIR modality)
            flair_volume = None
            for modality, path in local_paths.items():
                if modality == "flair":
                    import nibabel as nib
                    flair_volume = nib.load(str(path)).get_fdata(dtype=np.float32)
                    break
            
            slice_index = metadata["slice_index"]
            
            if flair_volume is not None:
                modality_arrays = {"flair": flair_volume}
                # Load other modalities for reference
                for modality in ["t1", "t1ce", "t2"]:
                    import nibabel as nib
                    vol = nib.load(str(local_paths[modality])).get_fdata(dtype=np.float32)
                    modality_arrays[modality] = vol
                original_pil = create_reference_image(modality_arrays, slice_index)
            else:
                # Fallback: create from model input
                original_pil = Image.new("RGB", (224, 224), color=(0, 0, 0))
            
            original_pil = original_pil.resize((224, 224))
            
            # ── RUN INFERENCE WITH GLOBAL MODEL ──
            model = _get_global_model()
            with torch.inference_mode():
                mask_tensor = model(image_tensor)
            
            # Generate XAI attribution with provenance
            xai_tensor, xai_provenance = generate_gradient_saliency(
                image_tensor, 
                model,
                model_checkpoint=_get_checkpoint_identifier(),
                model_version="armt-gan-2d-baseline",
            )
            
            # Compute segmentation alignment metric (model-attribution alignment analysis)
            # Uses the raw probability mask (before thresholding) for alignment
            alignment = compute_segmentation_alignment(
                attribution=xai_tensor,
                segmentation_mask=mask_tensor.cpu().numpy(),
                threshold=0.5,
            )
            
            logger.info(f"XAI alignment: inside={alignment['mean_inside']:.4f}, "
                       f"outside={alignment['mean_outside']:.4f}, "
                       f"ratio={alignment['ratio']}, "
                       f"tumor_pixels={alignment['tumor_pixel_count']}")
            
            # Convert to CPU numpy
            mask_cpu = mask_tensor.cpu().numpy()
            xai_cpu = xai_tensor
            
            # Generate visualization overlays
            seg_final_pil, xai_final_pil, tumor_detected, norm_mask = generate_clinical_overlays(
                original_pil=original_pil, 
                mask_tensor=mask_cpu, 
                xai_tensor=xai_cpu
            )
            
            # Compute max tumor probability from model output (raw sigmoid probability)
            max_tumor_probability = round(float(torch.sigmoid(mask_tensor).max().item()), 4)
            
            # Get or create ModelVersion for this checkpoint (needed for report and provenance)
            model_version = _get_or_create_model_version(
                db, _get_model_version_id(), _get_weights_path(), _get_model_version_config_hash(), _get_preprocessing_version()
            )

            # Save artifacts
            with tempfile.TemporaryDirectory() as tmpdirname:
                source_img_path = os.path.join(tmpdirname, f"{scan_id}_source.jpg")
                mask_path = os.path.join(tmpdirname, f"{scan_id}_mask.png")
                xai_path = os.path.join(tmpdirname, f"{scan_id}_xai.png")
                report_path = os.path.join(tmpdirname, f"{scan_id}_report.pdf")
                xai_raw_path = os.path.join(tmpdirname, f"{scan_id}_xai_raw.npy")
                
                # Save raw attribution as .npy for auditability
                np.save(xai_raw_path, xai_cpu)
                
                original_pil.save(source_img_path)
                seg_final_pil.save(mask_path)
                xai_final_pil.save(xai_path)
                
                generate_segmentation_report("PT-ANONYMIZED", scan_id, model_version.id, source_img_path, mask_path, xai_path, report_path)
                
                mask_obj_name = f"{scan_id}/mask.png"
                xai_obj_name = f"{scan_id}/xai.png"
                report_obj_name = f"{scan_id}/report.pdf"
                xai_raw_obj_name = f"{scan_id}/xai_raw.npy"
                
                generate_segmentation_report("PT-ANONYMIZED", scan_id, model_version.id, source_img_path, mask_path, xai_path, report_path)
                
                # Upload derived artifacts to MinIO
                upload_file_to_minio(mask_path, mask_obj_name)
                upload_file_to_minio(xai_path, xai_obj_name)
                upload_file_to_minio(report_path, report_obj_name)
                upload_file_to_minio(xai_raw_path, xai_raw_obj_name)
            
            # Update database with full provenance
            scan_record = db.query(Scan).filter(Scan.id == scan_id).first()
            if scan_record:
                scan_record.status = "SEGMENTED"
                scan_record.mask_path = mask_obj_name
                scan_record.xai_path = xai_obj_name
                scan_record.report_path = report_obj_name
                scan_record.xai_raw_path = xai_raw_obj_name  # Persist raw XAI object path
                
                # Create Prediction with model version linkage and nullable metrics
                # Note: dice/iou are nullable here since inference doesn't compute ground-truth metrics
                # They are populated by the evaluation pipeline when ground truth is available
                prediction = Prediction(
                    scan_id=scan_id,
                    model_version_id=model_version.id,
                    tumor_detected=tumor_detected,
                    max_tumor_probability=max_tumor_probability,
                    dice=None,  # Populated by evaluation pipeline when ground truth available
                    iou=None,   # Populated by evaluation pipeline when ground truth available
                )
                db.add(prediction)
                db.flush()  # Get prediction.id for Artifact FK
                
                # Create Artifact records as PENDING in the same transaction
                artifacts_to_create = [
                    Artifact(prediction_id=prediction.id, type="mask", object_path=mask_obj_name),
                    Artifact(prediction_id=prediction.id, type="xai", object_path=xai_obj_name),
                    Artifact(prediction_id=prediction.id, type="xai_raw", object_path=xai_raw_obj_name),
                    Artifact(prediction_id=prediction.id, type="report", object_path=report_obj_name),
                ]
                for artifact in artifacts_to_create:
                    db.add(artifact)
                db.flush()  # Get artifact IDs for transition
                
                # Transition each artifact from PENDING to COMPLETE after successful upload
                artifact_type_map = {
                    artifacts_to_create[0]: ("mask", mask_obj_name),
                    artifacts_to_create[1]: ("xai", xai_obj_name),
                    artifacts_to_create[2]: ("xai_raw", xai_raw_obj_name),
                    artifacts_to_create[3]: ("report", report_obj_name),
                }
                for artifact, (artifact_type, obj_name) in artifact_type_map.items():
                    result = transition_artifact_state(db, artifact.id, "COMPLETE")
                    if not result.success:
                        raise RuntimeError(f"Failed to transition {artifact_type} artifact {artifact.id} to COMPLETE: {result.reason}")
                
                db.commit()
            
            logger.info(f"Successfully processed 4-modality inference for ID: {scan_id}")
    
    except Exception as e:
        # Roll back the failed transaction first
        db.rollback()
        logger.error(f"Pipeline failed for Scan ID {scan_id}: {str(e)}")
        # Use a fresh, valid transaction to mark the Scan FAILED
        fresh_db: Session = SessionLocal()
        try:
            scan_record = fresh_db.query(Scan).filter(Scan.id == scan_id).first()
            if scan_record:
                scan_record.status = "FAILED"
                fresh_db.commit()
        finally:
            fresh_db.close()
    finally:
        db.close()