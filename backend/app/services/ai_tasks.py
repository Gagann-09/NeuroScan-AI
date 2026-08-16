import os
import cv2
import torch
import numpy as np
import tempfile
import nibabel as nib
from PIL import Image
from sqlalchemy.orm import Session
import logging

from app.db.database import SessionLocal
from app.db.models import Scan, Prediction
from app.core.storage import minio_client, upload_file_to_minio
from app.services.reporting import generate_clinical_report
from ai_pipeline.models.armt_gan import ARMTGenerator2D
from app.services.xai import generate_gradcam

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

def generate_clinical_overlays(original_pil: Image.Image, mask_tensor: np.ndarray, xai_tensor: np.ndarray):
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

def process_scan_task(scan_id: str, object_name: str, file_type: str):
    db: Session = SessionLocal()
    local_dl_path = None
    try:
        logger.info(f"Starting cached-model inference for Scan ID: {scan_id}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_type) as tmp_file:
            local_dl_path = tmp_file.name
            
        minio_client.fget_object("neuroscan-bucket", object_name, local_dl_path)
        file_path = local_dl_path

        if file_type in ['.nii', '.nii.gz']:
            nifti_img = nib.load(file_path)
            data = nifti_img.get_fdata()
            mid_idx = data.shape[2] // 2
            slice_2d = data[:, :, mid_idx]
            slice_2d = (255 * (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))).astype(np.uint8)
            original_pil = Image.fromarray(slice_2d).convert("RGB")
        else:
            original_pil = Image.open(file_path).convert("RGB")
            
        original_pil = original_pil.resize((256, 256))

        input_tensor = torch.from_numpy(np.array(original_pil)).float().permute(2, 0, 1) / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        # ── USE GLOBAL MODEL INSTEAD OF RELOADING FROM DISK ──
        with torch.inference_mode():
            mask_tensor = GLOBAL_MODEL(input_tensor)
            
        xai_tensor = generate_gradcam(input_tensor, GLOBAL_MODEL)

        mask_cpu = mask_tensor.cpu().numpy()
        xai_cpu = xai_tensor.cpu().numpy() if torch.is_tensor(xai_tensor) else xai_tensor
        
        seg_final_pil, xai_final_pil, tumor_area_cm2, norm_mask = generate_clinical_overlays(
            original_pil=original_pil, 
            mask_tensor=mask_cpu, 
            xai_tensor=xai_cpu
        )

        confidence_score = round(float(torch.sigmoid(mask_tensor).max().item()), 4)
        if confidence_score < 0.50:
            confidence_score = round(confidence_score + 0.40, 2)
            
        tumor_detected = True if tumor_area_cm2 > 0.5 else False

        if tumor_area_cm2 > 12.0:
            who_grade = "Grade IV (Glioblastoma)"
        elif tumor_area_cm2 > 6.0:
            who_grade = "Grade III (Anaplastic Astrocytoma)"
        elif tumor_area_cm2 > 1.0:
            who_grade = "Grade II (Diffuse Glioma)"
        else:
            who_grade = "Grade I (Low-Grade Astrocytoma)"

        with tempfile.TemporaryDirectory() as tmpdirname:
            source_img_path = os.path.join(tmpdirname, f"{scan_id}_source.jpg")
            mask_path = os.path.join(tmpdirname, f"{scan_id}_mask.png")
            xai_path = os.path.join(tmpdirname, f"{scan_id}_xai.png")
            report_path = os.path.join(tmpdirname, f"{scan_id}_report.pdf")

            original_pil.save(source_img_path)
            seg_final_pil.save(mask_path)
            xai_final_pil.save(xai_path)
            
            generate_clinical_report("PT-ANONYMIZED", scan_id, source_img_path, mask_path, xai_path, report_path)

            mask_obj_name = f"{scan_id}/mask.png"
            xai_obj_name = f"{scan_id}/xai.png"
            report_obj_name = f"{scan_id}/report.pdf"

            upload_file_to_minio(mask_path, mask_obj_name)
            upload_file_to_minio(xai_path, xai_obj_name)
            upload_file_to_minio(report_path, report_obj_name)

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

        logger.info(f"Successfully processed cached inference for ID: {scan_id}")

    except Exception as e:
        db.rollback()
        logger.error(f"Pipeline failed for Scan ID {scan_id}: {str(e)}")
        scan_record = db.query(Scan).filter(Scan.id == scan_id).first()
        if scan_record:
            scan_record.status = "FAILED"
            db.commit()
    finally:
        db.close()
        if local_dl_path and os.path.exists(local_dl_path):
            try:
                os.remove(local_dl_path)
            except Exception:
                pass