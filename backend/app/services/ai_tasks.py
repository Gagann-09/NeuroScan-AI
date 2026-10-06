import os
import cv2
import torch
import numpy as np
import tempfile
from PIL import Image
from sqlalchemy.orm import Session
import logging
from pathlib import Path

from app.db.database import SessionLocal
from app.db.models import Scan, ModalityFile, Prediction
from app.core.storage import minio_client, upload_file_to_minio
from app.services.reporting import generate_clinical_report
from ai_pipeline.models.armt_gan import ARMTGenerator2D
from app.services.xai import generate_gradient_saliency, compute_segmentation_alignment, XAIProvenance
from ai_pipeline.preprocessing import preprocess_brats_study, PreprocessingConfig

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../../ai_pipeline/weights/generator_latest.pth")

# ── LOAD MODEL ONCE GLOBALLY TO ELIMINATE REPEATED DISK I/O DELAYS ──
logger.info("Initializing ARMT-GAN model into global memory...")
GLOBAL_MODEL = ARMTGenerator2D().to(DEVICE)
if os.path.exists(WEIGHTS_PATH):
    GLOBAL_MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
GLOBAL_MODEL.eval()
logger.info("ARMT-GAN model successfully cached in memory.")

# Extract checkpoint identifier for provenance
CHECKPOINT_IDENTIFIER = os.path.basename(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else "unknown"


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
    estimated_area_cm2 = round(tumor_area_px * 0.11, 2)
    
    return seg_final_pil, xai_final_pil, estimated_area_cm2, norm_mask


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
            image_tensor = image_tensor.to(DEVICE)
            
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
            with torch.inference_mode():
                mask_tensor = GLOBAL_MODEL(image_tensor)
            
            # Generate XAI attribution with provenance
            xai_tensor, xai_provenance = generate_gradient_saliency(
                image_tensor, 
                GLOBAL_MODEL,
                model_checkpoint=CHECKPOINT_IDENTIFIER,
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
            seg_final_pil, xai_final_pil, tumor_area_cm2, norm_mask = generate_clinical_overlays(
                original_pil=original_pil, 
                mask_tensor=mask_cpu, 
                xai_tensor=xai_cpu
            )
            
            # Compute confidence score from model output (NO HEURISTIC BOOST)
            confidence_score = round(float(torch.sigmoid(mask_tensor).max().item()), 4)
            # REMOVED: if confidence_score < 0.50: confidence_score = round(confidence_score + 0.40, 2)
            
            tumor_detected = tumor_area_cm2 > 0.5
            
            # REMOVED: Heuristic WHO grading based on area
            # The model does NOT predict WHO grade - this is a research prototype
            who_grade = "Not predicted (research prototype)"
            
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
                
                generate_clinical_report("PT-ANONYMIZED", scan_id, source_img_path, mask_path, xai_path, report_path)
                
                mask_obj_name = f"{scan_id}/mask.png"
                xai_obj_name = f"{scan_id}/xai.png"
                report_obj_name = f"{scan_id}/report.pdf"
                xai_raw_obj_name = f"{scan_id}/xai_raw.npy"
                
                upload_file_to_minio(mask_path, mask_obj_name)
                upload_file_to_minio(xai_path, xai_obj_name)
                upload_file_to_minio(report_path, report_obj_name)
                upload_file_to_minio(xai_raw_path, xai_raw_obj_name)
            
            # Update database
            scan_record = db.query(Scan).filter(Scan.id == scan_id).first()
            if scan_record:
                scan_record.status = "SEGMENTED"
                scan_record.mask_path = mask_obj_name
                scan_record.xai_path = xai_obj_name
                scan_record.report_path = report_obj_name
                
                prediction = Prediction(
                    scan_id=scan_id,
                    tumor_detected=tumor_detected,
                    anomaly_area_cm2=tumor_area_cm2,
                    confidence_score=confidence_score,
                    who_grade=who_grade
                )
                db.add(prediction)
                db.commit()
            
            logger.info(f"Successfully processed 4-modality inference for ID: {scan_id}")
    
    except Exception as e:
        db.rollback()
        logger.error(f"Pipeline failed for Scan ID {scan_id}: {str(e)}")
        scan_record = db.query(Scan).filter(Scan.id == scan_id).first()
        if scan_record:
            scan_record.status = "FAILED"
            db.commit()
    finally:
        db.close()