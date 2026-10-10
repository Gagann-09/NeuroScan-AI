"""
NeuroScan AI - BraTS Preprocessing Pipeline

This is the SINGLE SOURCE OF TRUTH for BraTS preprocessing.
Both training and inference MUST use this module.

Contract:
- Input: 4 MRI modalities (T1, T1ce, T2, FLAIR) as NIfTI files
- Output: Tensor [1, 4, H, W] ready for ARMT-GAN generator
- Normalization: Z-score per modality on non-zero tissue
- Resizing: Bilinear interpolation to 224x224
- Slice selection: Same axial slice index across all modalities

Resource bounds (derived from BraTS dataset specifications and model constraints):
- BraTS volumes are typically 240x240x155 voxels (~9M voxels per modality)
- Max supported volume: 512x512x512 voxels (~134M voxels) to prevent OOM
- Max decompressed size per modality: 200MB (allows for compressed BraTS + margin)
- Supported data types: float32, int16, int32, uint8, uint16
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import nibabel as nib
import numpy as np
import torch
from PIL import Image


# Resource bounds (derived from BraTS dataset specifications and model constraints)
# Standard BraTS volumes: 240x240x155 voxels (~9M voxels/modality)
# Max supported volume: 512x512x512 voxels (~134M voxels) to prevent OOM
MAX_VOLUME_DIMENSION = 512
MAX_VOXEL_COUNT = 134_217_728  # 512^3
MAX_DECOMPRESSED_SIZE_MB = 200  # Max decompressed size per modality in MB
SUPPORTED_DTYPES = {np.float32, np.int16, np.int32, np.uint8, np.uint16, np.float64}

# BraTS modality ordering is FIXED and MUST match training
MODALITY_ORDER = ("t1", "t1ce", "t2", "flair")
MODALITY_SUFFIXES = {
    "t1": ("_t1.nii", "_t1.nii.gz", "-t1n.nii.gz"),
    "t1ce": ("_t1ce.nii", "_t1ce.nii.gz", "-t1c.nii.gz"),
    "t2": ("_t2.nii", "_t2.nii.gz", "-t2w.nii.gz"),
    "flair": ("_flair.nii", "_flair.nii.gz", "-t2f.nii.gz"),
}
SEGMENTATION_SUFFIXES = ("_seg.nii", "_seg.nii.gz", "-seg.nii.gz")

DEFAULT_IMAGE_SIZE = 224


@dataclass(frozen=True)
class PreprocessingConfig:
    """Immutable preprocessing configuration for reproducibility."""
    image_size: int = DEFAULT_IMAGE_SIZE
    modality_order: Tuple[str, ...] = MODALITY_ORDER
    normalize_nonzero: bool = True
    interpolation: str = "bilinear"  # bilinear for MRI, nearest for masks

    def to_dict(self) -> dict:
        return {
            "image_size": self.image_size,
            "modality_order": list(self.modality_order),
            "normalize_nonzero": self.normalize_nonzero,
            "interpolation": self.interpolation,
        }


class BraTSPreprocessor:
    """
    BraTS preprocessing pipeline.

    This class encapsulates all preprocessing steps to ensure
    identical behavior between training and inference.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def find_modality_file(self, patient_dir: Path, modality: str) -> Optional[Path]:
        """Find a single modality file in a patient directory."""
        suffixes = MODALITY_SUFFIXES.get(modality, ())
        for suffix in suffixes:
            matches = sorted(patient_dir.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        return None

    def find_segmentation_file(self, patient_dir: Path) -> Optional[Path]:
        """Find the segmentation file in a patient directory."""
        for suffix in SEGMENTATION_SUFFIXES:
            matches = sorted(patient_dir.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        return None

    def load_nifti(self, path: Path) -> np.ndarray:
        """Load a NIfTI file as float32 with validation."""
        image = nib.load(str(path))
        
        # Validate NIfTI header and data before loading full volume
        self._validate_nifti_image(image)
        
        return image.get_fdata(dtype=np.float32)

    def _validate_nifti_image(self, image: nib.Nifti1Image) -> None:
        """
        Validate NIfTI image header and data before loading full volume.
        Checks dimensions, data type, and estimated memory requirements.
        """
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


class BraTSPreprocessor:
    """
    BraTS preprocessing pipeline.

    This class encapsulates all preprocessing steps to ensure
    identical behavior between training and inference.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def find_modality_file(self, patient_dir: Path, modality: str) -> Optional[Path]:
        """Find a single modality file in a patient directory."""
        suffixes = MODALITY_SUFFIXES.get(modality, ())
        for suffix in suffixes:
            matches = sorted(patient_dir.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        return None

    def find_segmentation_file(self, patient_dir: Path) -> Optional[Path]:
        """Find the segmentation file in a patient directory."""
        for suffix in SEGMENTATION_SUFFIXES:
            matches = sorted(patient_dir.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        return None

    def load_nifti(self, path: Path) -> np.ndarray:
        """Load a NIfTI file as float32 with validation."""
        image = nib.load(str(path))
        
        # Validate NIfTI header and data before loading full volume
        self._validate_nifti_image(image)
        
        return image.get_fdata(dtype=np.float32)

    def _validate_nifti_image(self, image: nib.Nifti1Image) -> None:
        """
        Validate NIfTI image header and data before loading full volume.
        Checks dimensions, data type, and estimated memory requirements.
        """
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

    def normalize_nonzero(self, volume: np.ndarray) -> np.ndarray:
        """
        Z-score normalize non-zero tissue.
        Background (zero) remains zero.
        """
        volume = volume.astype(np.float32, copy=True)
        tissue = volume != 0
        
        if not np.any(tissue):
            return volume
        
        values = volume[tissue]
        mean = float(values.mean())
        std = float(values.std())
        
        if std > 1e-8:
            volume[tissue] = (values - mean) / std
        else:
            volume[tissue] = values - mean
        
        return volume

    def resize_slice(self, image: np.ndarray, target_size: int, is_mask: bool = False) -> np.ndarray:
        """
        Resize a single 2D slice.
        
        Args:
            image: 2D array (H, W)
            target_size: Target size for both dimensions
            is_mask: If True, use nearest-neighbor interpolation
        
        Returns:
            Resized 2D array (target_size, target_size)
        """
        # Convert to PIL for high-quality resizing
        if is_mask:
            pil_image = Image.fromarray(image.astype(np.uint8), mode="L")
            resample = Image.Resampling.NEAREST
        else:
            pil_image = Image.fromarray(image.astype(np.float32), mode="F")
            resample = Image.Resampling.BILINEAR
        
        pil_image = pil_image.resize(
            (target_size, target_size),
            resample=resample,
        )
        return np.asarray(pil_image, dtype=np.float32)

    def preprocess_modalities(
        self,
        modality_paths: dict[str, Path],
        slice_index: int,
    ) -> torch.Tensor:
        """
        Preprocess all 4 modalities for a given slice.
        
        Args:
            modality_paths: Dict mapping modality name to file path
            slice_index: Axial slice index to extract
        
        Returns:
            Tensor of shape [4, H, W] where H=W=image_size
        """
        modality_slices = []
        
        for modality in self.config.modality_order:
            path = modality_paths[modality]
            volume = self.load_nifti(path)
            
            if self.config.normalize_nonzero:
                volume = self.normalize_nonzero(volume)
            
            # Extract the same slice from all modalities
            slice_2d = volume[:, :, slice_index]
            
            # Resize to target size
            slice_resized = self.resize_slice(
                slice_2d,
                self.config.image_size,
                is_mask=False,
            )
            
            modality_slices.append(slice_resized)
        
        # Stack as [4, H, W]
        stacked = np.stack(modality_slices, axis=0)
        return torch.from_numpy(stacked).float()

    def preprocess_segmentation(
        self,
        segmentation_path: Path,
        slice_index: int,
    ) -> torch.Tensor:
        """
        Preprocess segmentation mask for a given slice.
        
        Args:
            segmentation_path: Path to segmentation NIfTI
            slice_index: Axial slice index to extract
        
        Returns:
            Tensor of shape [1, H, W] where H=W=image_size, binary {0, 1}
        """
        segmentation = self.load_nifti(segmentation_path)
        # BraTS labels: 0=background, 1=necrotic, 2=edema, 4=enhancing
        # Convert to binary tumor mask
        mask = (segmentation > 0).astype(np.float32)
        
        slice_2d = mask[:, :, slice_index]
        slice_resized = self.resize_slice(
            slice_2d,
            self.config.image_size,
            is_mask=True,
        )
        
        # Add channel dimension: [1, H, W]
        return torch.from_numpy(slice_resized).float().unsqueeze(0)


def preprocess_brats_study(
    patient_dir: Path,
    slice_index: Optional[int] = None,
    config: Optional[PreprocessingConfig] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
    """
    Preprocess a complete BraTS patient study.
    
    This is the MAIN ENTRY POINT for inference.
    
    Args:
        patient_dir: Directory containing 4 modalities + segmentation
        slice_index: Specific slice to use (None = middle slice with tumor)
        config: Preprocessing configuration
    
    Returns:
        Tuple of (image_tensor, mask_tensor_or_None, metadata)
        - image_tensor: [1, 4, H, W] ready for model
        - mask_tensor: [1, 1, H, W] binary mask, or None if no segmentation
        - metadata: Dict with preprocessing info for provenance
    """
    preprocessor = BraTSPreprocessor(config)
    
    # Find all modality files
    modality_paths = {}
    for modality in preprocessor.config.modality_order:
        path = preprocessor.find_modality_file(patient_dir, modality)
        if path is None:
            raise FileNotFoundError(
                f"Missing modality '{modality}' in {patient_dir}"
            )
        modality_paths[modality] = path
    
    # Find segmentation if available
    segmentation_path = preprocessor.find_segmentation_file(patient_dir)
    
    # Determine slice index if not provided
    if slice_index is None:
        if segmentation_path is not None:
            seg = preprocessor.load_nifti(segmentation_path)
            # Find middle slice with tumor
            tumor_slices = np.where(seg.sum(axis=(0, 1)) > 0)[0]
            if len(tumor_slices) > 0:
                slice_index = int(tumor_slices[len(tumor_slices) // 2])
            else:
                slice_index = seg.shape[2] // 2
        else:
            # No segmentation - use middle slice of first modality
            first_modality = preprocessor.load_nifti(
                modality_paths[preprocessor.config.modality_order[0]]
            )
            slice_index = first_modality.shape[2] // 2
    
    # Preprocess modalities
    image_tensor = preprocessor.preprocess_modalities(modality_paths, slice_index)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dim: [1, 4, H, W]
    
    # Preprocess segmentation if available
    mask_tensor = None
    if segmentation_path is not None:
        mask_tensor = preprocessor.preprocess_segmentation(segmentation_path, slice_index)
        mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dim: [1, 1, H, W]
    
    # Metadata for provenance
    metadata = {
        "patient_dir": str(patient_dir),
        "slice_index": slice_index,
        "modality_paths": {k: str(v) for k, v in modality_paths.items()},
        "segmentation_path": str(segmentation_path) if segmentation_path else None,
        "preprocessing_config": preprocessor.config.to_dict(),
    }
    
    return image_tensor, mask_tensor, metadata


def preprocess_brats_slice_tensor(
    modality_arrays: List[np.ndarray],
    config: Optional[PreprocessingConfig] = None,
) -> torch.Tensor:
    """
    Preprocess already-loaded modality arrays (for training dataset).
    
    Args:
        modality_arrays: List of 4 3D volumes in MODALITY_ORDER
        config: Preprocessing configuration
    
    Returns:
        Tensor of shape [4, H, W] for a single slice
    """
    preprocessor = BraTSPreprocessor(config)
    
    modality_slices = []
    for volume in modality_arrays:
        if preprocessor.config.normalize_nonzero:
            volume = preprocessor.normalize_nonzero(volume)
        # Caller handles slice extraction
        slice_2d = volume  # Expected to already be 2D slice
        slice_resized = preprocessor.resize_slice(slice_2d, preprocessor.config.image_size, is_mask=False)
        modality_slices.append(slice_resized)
    
    stacked = np.stack(modality_slices, axis=0)
    return torch.from_numpy(stacked).float()