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
import nibabel as nib

from app.db.database import SessionLocal
from app.db.models import Scan, ModalityFile, Prediction, ModelVersion, Artifact
from app.core.storage import minio_client, upload_file_to_minio
from app.services.reporting import generate_segmentation_report
from ai_pipeline.models.armt_gan import ARMTGenerator2D
from app.services.xai import generate_gradient_saliency, compute_segmentation_alignment, XAIProvenance
from ai_pipeline.preprocessing import preprocess_brats_study, PreprocessingConfig
from ai_pipeline.preprocessing.brats_preprocessing import (
    BraTSPreprocessor,
    MAX_VOLUME_DIMENSION,
    MAX_VOXEL_COUNT,
    MAX_DECOMPRESSED_SIZE_MB,
    SUPPORTED_DTYPES,
)

logger = logging.getLogger(__name__)


def _validate_nifti_file(path: Path) -> None:
    """
    Validate a NIfTI file on disk before processing.
    Checks dimensions, data type, and estimated memory requirements.
    Raises ValueError if validation fails.
    """
    image = nib.load(str(path))
    
    # Check header magic and type
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError(f"Unsupported image type: {type(image).__name__}. Only NIfTI-1 and NIfTI-2 supported.")
    
    # Validate header magic
    magic = image.header.get('magic', b'')
    if magic not in (b'ni1', b'n+1'):
        raise ValueError(f"Invalid NIfTI magic: {magic!r}")
    
    # Check data type
    dtype = image.get_data_dtype()
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported data type: {dtype}. "
            f"Supported types: {', '.join(str(dt) for dt in SUPPORTED_DTYPES)}"
        )
    
    # Check dimensions
    shape = image.shape
    if len(shape) < 3:
        raise ValueError(f"Expected 3D or 4D volume, got shape {shape}")
    
    # Check dimension bounds
    for i, dim in enumerate(shape[:3]):
        if dim <= 0:
            raise ValueError(f"Invalid dimension {i}: {dim} (must be > 0)")
        if dim > MAX_VOLUME_DIMENSION:
            raise ValueError(
                f"Dimension {i} ({dim}) exceeds maximum allowed ({MAX_VOLUME_DIMENSION})"
            )
    
    # Check total voxel count
    voxel_count = np.prod(shape[:3])
    if voxel_count > MAX_VOXEL_COUNT:
        raise ValueError(
            f"Volume voxel count ({voxel_count:,}) exceeds maximum allowed ({MAX_VOXEL_COUNT:,})"
        )
    
    # Estimate decompressed size and check against limit
    dtype_size = np.dtype(image.get_data_dtype()).itemsize
    estimated_size_mb = (voxel_count * dtype_size) / (1024 * 1024)
    if estimated_size_mb > MAX_DECOMPRESSED_SIZE_MB:
        raise ValueError(
            f"Estimated decompressed size ({estimated_size_mb:.1f} MB) "
            f"exceeds maximum allowed ({MAX_DECOMPRESSED_SIZE_MB} MB)"
        )


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
    uploaded_derived: list[str] = []
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
            
            # Validate downloaded NIfTI files before processing
            for modality, path in local_paths.items():
                _validate_nifti_file(path)
            
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
                mask_path = os.path.join(tmpdir, f"{scan_id}_mask.png")
                xai_path = os.path.join(tmpdir, f"{scan_id}_xai.png")
                report_path = os.path.join(tmpdir, f"{scan_id}_report.pdf")
                xai_raw_path = os.path.join(tmpdir, f"{scan_id}_xai_raw.npy")
                
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
                
                mask_obj_name = f"{scan_id}/mask.png"
                xai_obj_name = f"{scan_id}/xai.png"
                report_obj_name = f"{scan_id}/report.pdf"
                xai_raw_obj_name = f"{scan_id}/xai_raw.npy"
                
                generate_segmentation_report("PT-ANONYMIZED", scan_id, model_version.id, source_img_path, mask_path, xai_path, report_path)
                
                # Upload derived artifacts to MinIO, tracking successful uploads for compensation
                uploaded_derived: list[str] = []
                bucket_name = "neuroscan-bucket"
                
                upload_file_to_minio(mask_path, mask_obj_name)
                uploaded_derived.append(mask_obj_name)
                
                upload_file_to_minio(xai_path, xai_obj_name)
                uploaded_derived.append(xai_obj_name)
                
                upload_file_to_minio(report_path, report_obj_name)
                uploaded_derived.append(report_obj_name)
                
                upload_file_to_minio(xai_raw_path, xai_raw_obj_name)
                uploaded_derived.append(xai_raw_obj_name)
            
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
        # Best-effort cleanup of derived objects uploaded during this attempt
        if uploaded_derived:
            _cleanup_derived_objects("neuroscan-bucket", uploaded_derived)
        # Use a fresh, valid transaction to mark the Scan FAILED
        fresh_db: Session = SessionLocal()
        try:
            scan_record = fresh_db.query(Scan).filter(Scan.id == scan_id).first()
            if scan_record:
                scan_record.status = "FAILED"
                fresh_db.commit()
        finally:
            fresh_db.close()
        db.close()