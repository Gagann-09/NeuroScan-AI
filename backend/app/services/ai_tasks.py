import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import tempfile
import torch
from PIL import Image
import numpy as np
import nibabel as nib

from app.core.celery_app import celery_app
from app.db.database import SessionLocal
from app.db.models import MRIScan, Patient
from app.core.storage import minio_client

from ai_pipeline.preprocessing.spatial_2d.kaggle_prep import preprocess_image
from ai_pipeline.models.armt_gan import ARMTGenerator2D
from app.services.xai import generate_xai_heatmap
from app.services.reporting import generate_clinical_report

print("[AI Worker] Loading ARMT-GAN Generator into Memory...")
armt_model = ARMTGenerator2D()
armt_model.eval() 

def process_nifti_to_tensor_and_image(nii_path, save_orig_img_path):
    """Extracts a 2D middle slice from a 3D NIfTI volume for inference and visual reporting."""
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()
    
    # Handle 3D or 4D NIfTI shapes safely
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
        
    mid_idx = data.shape[2] // 2
    mid_slice = data[:, :, mid_idx]
    
    mid_slice = np.rot90(mid_slice)
    normalized = np.clip(mid_slice, 0, np.max(mid_slice))
    if np.max(normalized) > 0:
        normalized = (normalized / np.max(normalized)) * 255
        
    img_pil = Image.fromarray(normalized.astype(np.uint8)).convert("RGB").resize((224, 224))
    img_pil.save(save_orig_img_path)

    tensor = torch.tensor(np.array(img_pil), dtype=torch.float32).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor

@celery_app.task(name="process_mri_scan")
def process_mri_scan(scan_id: str):
    db = SessionLocal()
    scan = db.query(MRIScan).filter(MRIScan.id == scan_id).first()

    if not scan:
        db.close()
        return {"status": "error", "message": "Scan not found"}

    scan.status = "PROCESSING"
    db.commit()

    try:
        bucket_name, object_name = scan.file_path.split("/", 1)
        ext = object_name.split(".")[-1].lower()
        folder_prefix = object_name.rsplit('/', 1)[0]

        # Download original scan
        tmp_orig = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        tmp_orig.close() 
        minio_client.fget_object(bucket_name, object_name, tmp_orig.name)

        if scan.file_type == "IMAGE" or ext in ["jpg", "jpeg", "png"]:
            tensor = preprocess_image(tmp_orig.name)
            input_tensor = tensor.unsqueeze(0)
            inference_orig_path = tmp_orig.name
        else:
            tmp_slice_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp_slice_img.close()
            input_tensor = process_nifti_to_tensor_and_image(tmp_orig.name, tmp_slice_img.name).unsqueeze(0)
            inference_orig_path = tmp_slice_img.name

        with torch.no_grad():
            mask_tensor = armt_model(input_tensor)
        
        # 1. Export Raw Mask to PNG
        mask_np = mask_tensor[0, 0].cpu().numpy() * 255
        mask_img = Image.fromarray(mask_np.astype(np.uint8))
        tmp_mask = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_mask.close()
        mask_img.save(tmp_mask.name)

        # 2. Generate XAI Heatmap
        tmp_xai = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_xai.close()
        generate_xai_heatmap(input_tensor, mask_tensor, tmp_xai.name)

        # 3. Generate Clinical PDF Report
        patient = db.query(Patient).filter(Patient.id == scan.patient_id).first()
        pt_id = patient.patient_identifier if patient else "UNKNOWN"
        
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_pdf.close()
        generate_clinical_report(pt_id, scan_id, inference_orig_path, tmp_mask.name, tmp_xai.name, tmp_pdf.name)

        # 4. Upload all Clinical Assets to MinIO
        minio_client.fput_object(bucket_name, f"{folder_prefix}/mask_{scan_id}.png", tmp_mask.name)
        minio_client.fput_object(bucket_name, f"{folder_prefix}/xai_{scan_id}.png", tmp_xai.name)
        minio_client.fput_object(bucket_name, f"{folder_prefix}/report_{scan_id}.pdf", tmp_pdf.name)

        # 5. Cleanup Local Storage
        for p in [tmp_mask.name, tmp_xai.name, tmp_pdf.name, tmp_orig.name]:
            if os.path.exists(p):
                os.remove(p)
        if 'tmp_slice_img' in locals() and os.path.exists(tmp_slice_img.name):
            os.remove(tmp_slice_img.name)

        scan.status = "SEGMENTED"
        db.commit()

        print(f"[AI Worker] SUCCESS | Scan: {scan_id} | Clinical Report Generated & Saved to MinIO.")
        return {"status": "success", "mask_shape": list(mask_tensor.shape)}

    except Exception as e:
        scan.status = "FAILED"
        db.commit()
        import traceback
        traceback.print_exc()
        print(f"[AI Worker] Failed processing scan {scan_id}: {str(e)}")
        return {"status": "error", "message": str(e)}
    finally:
        db.close()