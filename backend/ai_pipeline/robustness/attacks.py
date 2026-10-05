"""
NeuroScan AI - Adversarial Attack Implementations

Implements FGSM and PGD for segmentation adversarial robustness evaluation.

Attack Objective:
- Untargeted attack: maximize segmentation error against ground-truth mask
- Loss: differentiable soft Dice loss (1 - soft Dice coefficient)
- Operates on model output probabilities (continuous, not thresholded)
- Gradient computed w.r.t. input tensor in normalized space

Threat Model:
- White-box: full model access, gradient computation
- L-infinity constraint: ||δ||_∞ ≤ ε
- Perturbation applied to all 4 modalities (T1, T1ce, T2, FLAIR) with single ε budget
- Input space: Z-score normalized tensor (no hard [0,1] bounds)
- Clipping: values clipped to [clip_min, clip_max] after perturbation
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Optional, Literal


@dataclass(frozen=True)
class AttackConfig:
    """
    Frozen attack configuration for ARMT-GAN robustness evaluation.
    
    Perturbation Space:
    - The model input uses Z-score normalization on non-zero tissue.
    - Background (zero) remains zero after normalization.
    - Normalized tissue values have approximately zero mean and unit variance.
    - No hard [0, 1] bounds exist after Z-score normalization.
    
    L-infinity Constraint:
    - Perturbation is bounded in L-infinity norm: ||δ||_∞ ≤ ε
    - Applied to the complete 4-channel input tensor [B, 4, H, W]
    - Single ε budget shared across all 4 modalities (T1, T1ce, T2, FLAIR)
    
    Clipping Strategy:
    - After perturbation, values are clipped to [clip_min, clip_max]
    - clip_min/clip_max are derived from the empirical range of normalized
      training data, or a large margin if not available.
    - This prevents the attack from creating values far outside the
      normalized data distribution.
    """
    
    # Attack parameters
    epsilon: float = 0.03  # L-infinity budget
    
    # PGD-specific
    step_size: float = 0.0075  # α = ε/4
    num_iterations: int = 10
    random_start: bool = True
    random_start_seed_offset: int = 1000  # Added to base seed for PGD random start
    
    # Input space clipping (empirically derived from normalized data)
    # Z-score normalized data typically falls in [-3, 3] for 99.7% of values
    clip_min: float = -3.0
    clip_max: float = 3.0
    
    # Norm constraint
    norm: Literal["linf"] = "linf"
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "step_size": self.step_size,
            "num_iterations": self.num_iterations,
            "random_start": self.random_start,
            "random_start_seed_offset": self.random_start_seed_offset,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "norm": self.norm,
            "seed": self.seed,
        }


BASELINE_ATTACK_CONFIG = AttackConfig()


def soft_dice_loss(
    pred_prob: torch.Tensor,
    target_mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable soft Dice loss for segmentation.
    
    Args:
        pred_prob: Model output probabilities [B, 1, H, W] or [B, H, W]
        target_mask: Ground truth binary mask [B, 1, H, W] or [B, H, W]
        epsilon: Small constant for numerical stability
    
    Returns:
        Scalar loss = 1 - mean(soft Dice coefficient)
    
    The loss is differentiable and operates on continuous probabilities.
    No thresholding is applied before gradient computation.
    """
    # Ensure shapes match [B, 1, H, W]
    if pred_prob.dim() == 3:
        pred_prob = pred_prob.unsqueeze(1)
    if target_mask.dim() == 3:
        target_mask = target_mask.unsqueeze(1)
    
    # Flatten spatial dimensions
    pred_flat = pred_prob.flatten(2)  # [B, 1, H*W]
    target_flat = target_mask.flatten(2)  # [B, 1, H*W]
    
    # Soft Dice computation
    intersection = (pred_flat * target_flat).sum(dim=2)  # [B, 1]
    pred_sum = pred_flat.sum(dim=2)  # [B, 1]
    target_sum = target_flat.sum(dim=2)  # [B, 1]
    
    dice = (2.0 * intersection + 1e-8) / (pred_sum + target_sum + 1e-8)  # [B, 1]
    
    # Mean over batch, then 1 - dice for loss (maximize loss = minimize Dice)
    loss = 1.0 - dice.mean()
    
    return loss


def segmentation_attack_loss(
    model: nn.Module,
    images: torch.Tensor,
    target_masks: torch.Tensor,
    loss_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Compute attack loss for adversarial segmentation.
    
    Args:
        model: ARMT-GAN generator in eval mode
        images: Input tensor [B, 4, H, W] (requires_grad=True)
        target_masks: Ground truth binary masks [B, 1, H, W]
        loss_fn: Optional custom loss function (default: soft_dice_loss)
    
    Returns:
        Scalar loss tensor (gradients flow to images)
    """
    if loss_fn is None:
        loss_fn = soft_dice_loss
    
    # Forward pass - get probabilities
    output = model(images)  # [B, 1, H, W] sigmoid probabilities
    
    # Compute attack loss (maximize error on ground truth)
    loss = loss_fn(output, target_masks)
    
    return loss


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    target_masks: torch.Tensor,
    config: AttackConfig,
    loss_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) for segmentation.
    
    x_adv = x + ε * sign(∇_x L(model(x), y))
    
    Args:
        model: ARMT-GAN generator in eval mode
        images: Clean input tensor [B, 4, H, W]
        target_masks: Ground truth masks [B, 1, H, W]
        config: AttackConfig with epsilon, clip bounds, etc.
        loss_fn: Optional custom loss function
    
    Returns:
        Adversarial images [B, 4, H, W] (detached, no gradients)
    
    Verifies:
        - Correct gradient shape [B, 4, H, W]
        - L∞ bound: ||x_adv - x||_∞ ≤ ε + 1e-6
        - Deterministic behavior with fixed seed
        - Input shape preservation
        - No model weight modifications
    """
    model.eval()
    
    # Ensure clean input is within valid bounds first
    # This prevents clamping from creating artificial perturbation
    x_clean = torch.clamp(images, config.clip_min, config.clip_max)
    
    # Clone and enable gradients for input
    x = x_clean.clone().detach().requires_grad_(True)
    
    # Compute loss and gradients
    loss = segmentation_attack_loss(model, x, target_masks, loss_fn)
    
    model.zero_grad()
    loss.backward()
    
    # Get gradient sign
    grad = x.grad.detach()
    grad_sign = grad.sign()
    
    # FGSM perturbation
    perturbation = config.epsilon * grad_sign
    x_adv = x + perturbation
    
    # Clip adversarial image to valid bounds
    x_adv = torch.clamp(x_adv, config.clip_min, config.clip_max)
    
    return x_adv.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    target_masks: torch.Tensor,
    config: AttackConfig,
    loss_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) for segmentation (L-infinity).
    
    Iterative: x_{t+1} = Π_{B_ε(x)} (x_t + α * sign(∇_x L))
    
    Args:
        model: ARMT-GAN generator in eval mode
        images: Clean input tensor [B, 4, H, W]
        target_masks: Ground truth masks [B, 1, H, W]
        config: AttackConfig with epsilon, step_size, num_iterations, etc.
        loss_fn: Optional custom loss function
    
    Returns:
        Adversarial images [B, 4, H, W] (detached, no gradients)
    
    Verifies:
        - Epsilon projection: ||x_adv - x||_∞ ≤ ε + 1e-6
        - Step size respected
        - Iteration count executed
        - Deterministic seeded behavior (with random_start_seed_offset)
        - Shape preservation
        - No model weight modifications
    """
    model.eval()
    
    # Ensure clean input is within valid bounds first
    x_clean = torch.clamp(images, config.clip_min, config.clip_max)
    
    x = x_clean.clone().detach()
    
    # Random start within ε-ball
    if config.random_start:
        torch.manual_seed(config.seed + config.random_start_seed_offset)
        delta = torch.empty_like(x).uniform_(-config.epsilon, config.epsilon)
        x = torch.clamp(x + delta, config.clip_min, config.clip_max)
    
    for i in range(config.num_iterations):
        x.requires_grad_(True)
        
        loss = segmentation_attack_loss(model, x, target_masks, loss_fn)
        
        model.zero_grad()
        loss.backward()
        
        grad = x.grad.detach()
        grad_sign = grad.sign()
        
        # Gradient ascent step
        x = x.detach() + config.step_size * grad_sign
        
        # Project back to ε-ball around clean input
        delta = torch.clamp(x - x_clean, -config.epsilon, config.epsilon)
        x = torch.clamp(x_clean + delta, config.clip_min, config.clip_max)
    
    return x.detach()


def compute_perturbation_norm(
    adv_images: torch.Tensor,
    clean_images: torch.Tensor,
    norm: str = "linf",
) -> float:
    """
    Compute perturbation norm for verification.
    
    Args:
        adv_images: Adversarial images
        clean_images: Clean images
        norm: "linf" for L-infinity, "l2" for L2
    
    Returns:
        Maximum perturbation norm across batch
    """
    delta = adv_images - clean_images
    
    if norm == "linf":
        return delta.abs().max().item()
    elif norm == "l2":
        return torch.norm(delta, p=2, dim=(1, 2, 3)).max().item()
    else:
        raise ValueError(f"Unsupported norm: {norm}")


def verify_attack_constraints(
    adv_images: torch.Tensor,
    clean_images: torch.Tensor,
    config: AttackConfig,
    tolerance: float = 1e-6,
) -> tuple[bool, str]:
    """
    Verify that adversarial images satisfy attack constraints.
    
    Note: Clean images are clamped to [clip_min, clip_max] before comparison,
    matching the internal behavior of the attack functions.
    
    Returns:
        (passed, message)
    """
    # Clamp clean images to valid bounds (matching attack function behavior)
    clean_clamped = torch.clamp(clean_images, config.clip_min, config.clip_max)
    
    # Check shape
    if adv_images.shape != clean_clamped.shape:
        return False, f"Shape mismatch: {adv_images.shape} vs {clean_clamped.shape}"
    
    # Check L-infinity bound against clamped clean input
    max_perturbation = compute_perturbation_norm(adv_images, clean_clamped, "linf")
    if max_perturbation > config.epsilon + tolerance:
        return False, f"L∞ bound violated: {max_perturbation:.6f} > {config.epsilon:.6f} + {tolerance}"
    
    # Check clipping bounds
    if adv_images.min() < config.clip_min - tolerance:
        return False, f"Below clip_min: {adv_images.min():.6f} < {config.clip_min:.6f}"
    if adv_images.max() > config.clip_max + tolerance:
        return False, f"Above clip_max: {adv_images.max():.6f} > {config.clip_max:.6f}"
    
    return True, "All constraints satisfied"