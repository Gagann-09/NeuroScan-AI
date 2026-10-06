import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class XAIProvenance:
    """
    Provenance metadata for gradient-based input saliency attribution.
    
    This records the exact method configuration for auditability.
    """
    method: str = "gradient-based input saliency"
    target: str = "sum of tumor logits across spatial dimensions"
    normalization: str = "min-max [0, 1]"
    aggregation: str = "max absolute gradient across input channels"
    model_checkpoint: Optional[str] = None
    model_version: Optional[str] = None
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    
    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "target": self.target,
            "normalization": self.normalization,
            "aggregation": self.aggregation,
            "model_checkpoint": self.model_checkpoint,
            "model_version": self.model_version,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shape": list(self.output_shape) if self.output_shape else None,
        }


def generate_gradient_saliency(
    input_tensor: torch.Tensor,
    model: torch.nn.Module,
    model_checkpoint: Optional[str] = None,
    model_version: Optional[str] = None,
) -> tuple[np.ndarray, XAIProvenance]:
    """
    Computes gradient-based input saliency map (NOT Grad-CAM).
    
    This method computes the gradient of the output with respect to the input,
    producing a spatial attribution map showing which input regions most
    influence the segmentation prediction.
    
    Method: Input-gradient saliency (also known as sensitivity analysis)
    Target: Sum of output logits across spatial dimensions (tumor probability mass)
    Normalization: Min-max to [0, 1]
    Channel aggregation: Max of absolute gradients across input channels
    
    This is NOT Grad-CAM, which requires a specific target layer and
    computes gradients with respect to feature maps, not input pixels.
    
    Args:
        input_tensor: Input tensor of shape [B, 4, H, W]
        model: ARMT-GAN generator in eval mode
        model_checkpoint: Optional path/identifier of model checkpoint
        model_version: Optional model version identifier
        
    Returns:
        Tuple of (saliency_map, provenance) where:
        - saliency_map: numpy array of shape [B, 1, H, W] with values in [0, 1]
        - provenance: XAIProvenance dataclass with method metadata
        
    Raises:
        RuntimeError: If gradients are unavailable (e.g., model not differentiable)
        ValueError: If input tensor shape is not [B, 4, H, W]
    """
    if input_tensor.dim() != 4 or input_tensor.shape[1] != 4:
        raise ValueError(f"Expected input shape [B, 4, H, W], got {input_tensor.shape}")
    
    model.eval()
    # Clone and enable gradients for input
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass through ARMT-GAN generator
    output = model(input_tensor)
    
    # Compute attribution gradients with respect to the output prediction
    # Target: sum of tumor logits across all spatial positions
    score = output.sum()
    score.backward()
    
    if input_tensor.grad is None:
        raise RuntimeError(
            "Gradients unavailable: input tensor did not receive gradients. "
            "Ensure the model is differentiable and input requires_grad=True."
        )
    
    grads = input_tensor.grad.detach()
    
    # Generate spatial saliency attribution map across input channels
    # Take max of absolute gradients across the 4 modality channels
    saliency, _ = torch.max(torch.abs(grads), dim=1, keepdim=True)
    
    # Normalize between 0 and 1 using min-max
    saliency_min = saliency.min()
    saliency_max = saliency.max()
    
    if saliency_max > saliency_min:
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min)
    else:
        # Flat gradient case: all values identical (typically all zeros)
        # Return zero map explicitly rather than dividing by zero
        saliency = torch.zeros_like(saliency)
    
    saliency_np = saliency.cpu().detach().numpy()
    
    # Build provenance record
    provenance = XAIProvenance(
        model_checkpoint=model_checkpoint,
        model_version=model_version,
        input_shape=tuple(input_tensor.shape),
        output_shape=tuple(saliency_np.shape),
    )
    
    return saliency_np, provenance


def compute_segmentation_alignment(
    attribution: np.ndarray,
    segmentation_mask: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute quantitative alignment between attribution map and segmentation.
    
    Measures mean attribution inside vs outside the predicted tumor region.
    This is a model-attribution alignment analysis, NOT clinical validation.
    
    Args:
        attribution: Attribution map [B, 1, H, W] or [H, W], values in [0, 1]
        segmentation_mask: Probability mask [B, 1, H, W] or [H, W], values in [0, 1]
        threshold: Threshold to binarize segmentation mask (default 0.5)
        
    Returns:
        Dict with alignment metrics:
        - mean_inside: Mean attribution inside tumor region
        - mean_outside: Mean attribution outside tumor region
        - ratio: mean_inside / mean_outside (inf if mean_outside == 0)
        - tumor_pixel_count: Number of pixels in tumor region
        - background_pixel_count: Number of pixels in background region
    """
    # Squeeze to [H, W] for computation
    attr = attribution.squeeze()
    mask = segmentation_mask.squeeze()
    
    if attr.shape != mask.shape:
        raise ValueError(f"Shape mismatch: attribution {attr.shape} vs mask {mask.shape}")
    
    # Binarize segmentation mask
    binary_mask = (mask > threshold).astype(bool)
    
    tumor_pixels = binary_mask.sum()
    background_pixels = (~binary_mask).sum()
    
    if tumor_pixels == 0:
        # Empty tumor mask: no tumor region to measure
        return {
            "mean_inside": 0.0,
            "mean_outside": float(attr[~binary_mask].mean()) if background_pixels > 0 else 0.0,
            "ratio": 0.0,
            "tumor_pixel_count": 0,
            "background_pixel_count": int(background_pixels),
            "note": "Empty tumor mask - no alignment measurable",
        }
    
    mean_inside = float(attr[binary_mask].mean())
    mean_outside = float(attr[~binary_mask].mean()) if background_pixels > 0 else 0.0
    
    ratio = mean_inside / mean_outside if mean_outside > 0 else float('inf')
    
    return {
        "mean_inside": mean_inside,
        "mean_outside": mean_outside,
        "ratio": ratio if ratio != float('inf') else None,
        "tumor_pixel_count": int(tumor_pixels),
        "background_pixel_count": int(background_pixels),
    }


# Backward compatibility alias (to be removed)
def generate_gradcam(input_tensor, model):
    """DEPRECATED: This is gradient-based saliency, not Grad-CAM."""
    import warnings
    warnings.warn(
        "generate_gradcam is deprecated. Use generate_gradient_saliency. "
        "This method computes input-gradient saliency, not Grad-CAM.",
        DeprecationWarning,
        stacklevel=2
    )
    # Call new function but discard provenance for backward compat
    saliency, _ = generate_gradient_saliency(input_tensor, model)
    return saliency