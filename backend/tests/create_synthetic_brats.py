"""
Synthetic BraTS Data Generator for Smoke Testing

Creates a minimal BraTS-like dataset for verifying the training pipeline
without requiring real BraTS data.
"""

import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np


def create_synthetic_brats_dataset(num_patients: int = 4, output_dir: Path = None) -> Path:
    """
    Create a synthetic BraTS dataset for smoke testing.
    
    Args:
        num_patients: Number of patients to create (must be >= 2 for train/val split)
        output_dir: Directory to create dataset in (None = temp directory)
    
    Returns:
        Path to the created dataset directory
    """
    if num_patients < 2:
        raise ValueError("Need at least 2 patients for train/validation split")
    
    if output_dir is None:
        tmpdir = tempfile.mkdtemp(prefix="synthetic_brats_")
        output_dir = Path(tmpdir)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Modality base intensities (to make them distinguishable)
    base_values = {
        "t1": 500,
        "t1ce": 800,
        "t2": 1200,
        "flair": 1500,
    }
    
    tumor_boost = {
        "t1": 200,
        "t1ce": 300,
        "t2": 400,
        "flair": 500,
    }
    
    # Standard BraTS-like dimensions (smaller for speed)
    shape = (64, 64, 32)
    
    for p in range(num_patients):
        patient_id = f"BraTS20_Training_{p:03d}"
        patient_dir = output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Use different random seed per patient for variety
        np.random.seed(42 + p)
        
        for modality in ["t1", "t1ce", "t2", "flair"]:
            volume = (np.random.randn(*shape).astype(np.float32) * 50 + base_values[modality])
            
            # Add tumor region in middle slices
            tumor_z_start, tumor_z_end = 10, 22
            tumor_y_start, tumor_y_end = 20, 44
            tumor_x_start, tumor_x_end = 20, 44
            
            volume[tumor_y_start:tumor_y_end, tumor_x_start:tumor_x_end, tumor_z_start:tumor_z_end] += tumor_boost[modality]
            
            # Save as NIfTI
            img = nib.Nifti1Image(volume, affine=np.eye(4))
            nib.save(img, patient_dir / f"{patient_id}_{modality}.nii.gz")
        
        # Create segmentation (binary tumor mask)
        seg = np.zeros(shape, dtype=np.float32)
        seg[tumor_y_start:tumor_y_end, tumor_x_start:tumor_x_end, tumor_z_start:tumor_z_end] = 1.0
        
        seg_img = nib.Nifti1Image(seg, affine=np.eye(4))
        nib.save(seg_img, patient_dir / f"{patient_id}_seg.nii.gz")
    
    print(f"Created synthetic BraTS dataset at: {output_dir}")
    print(f"Patients: {[d.name for d in output_dir.iterdir() if d.is_dir()]}")
    
    return output_dir


if __name__ == "__main__":
    create_synthetic_brats_dataset(num_patients=4)