# ai_pipeline/preprocessing/volumetric_3d/brats_prep.py
import nibabel as nib
import numpy as np
import torch

def preprocess_nifti(local_file_path: str) -> torch.Tensor:
    """
    Loads a 3D NIfTI file (.nii), extracts voxel data, applies Z-score 
    normalization to non-background tissue, and converts to a PyTorch tensor.
    """
    img = nib.load(local_file_path)
    data = img.get_fdata(dtype=np.float32)

    # Z-score normalization strictly on brain tissue (> 0)
    brain_mask = data > 0
    if brain_mask.any():
        mean = np.mean(data[brain_mask])
        std = np.std(data[brain_mask])
        if std > 0:
            data[brain_mask] = (data[brain_mask] - mean) / std

    # Convert to PyTorch Tensor and add channel dimension (1, Depth, Height, Width)
    tensor = torch.from_numpy(data).float()
    tensor = tensor.unsqueeze(0)
    return tensor