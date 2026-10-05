"""
NeuroScan AI - Shared Preprocessing Module

This module provides the canonical preprocessing pipeline used by both
training and inference. It ensures parity between the two paths.

Contract:
- Input: 4 MRI modalities (T1, T1ce, T2, FLAIR) as NIfTI files
- Output: Tensor [1, 4, 224, 224] ready for ARMT-GAN generator
- Normalization: Z-score per modality on non-zero tissue
- Resizing: Bilinear interpolation to 224x224
- Slice selection: Same axial slice index across all modalities
"""

from .brats_preprocessing import (
    BraTSPreprocessor,
    PreprocessingConfig,
    preprocess_brats_study,
    preprocess_brats_slice_tensor,
)

__all__ = [
    "BraTSPreprocessor",
    "PreprocessingConfig",
    "preprocess_brats_study",
    "preprocess_brats_slice_tensor",
]