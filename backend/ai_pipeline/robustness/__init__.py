"""
NeuroScan AI - Robustness Module

Provides adversarial robustness evaluation for ARMT-GAN.
"""

from ai_pipeline.robustness.config import (
    AttackConfig,
    BASELINE_ATTACK_CONFIG,
    RobustnessConfig,
    BASELINE_ROBUSTNESS_CONFIG,
)
from ai_pipeline.robustness.attacks import (
    soft_dice_loss,
    segmentation_attack_loss,
    fgsm_attack,
    pgd_attack,
    compute_perturbation_norm,
    verify_attack_constraints,
)
from ai_pipeline.robustness.evaluate_robustness import (
    run_robustness_evaluation,
    print_robustness_result,
    save_robustness_result,
    RobustnessResult,
    AdversarialMetrics,
    BASELINE_ATTACK_CONFIG as BASELINE_ATTACK_CONFIG_EVAL,
)

__all__ = [
    "AttackConfig",
    "BASELINE_ATTACK_CONFIG",
    "RobustnessConfig",
    "BASELINE_ROBUSTNESS_CONFIG",
    "soft_dice_loss",
    "segmentation_attack_loss",
    "fgsm_attack",
    "pgd_attack",
    "compute_perturbation_norm",
    "verify_attack_constraints",
    "run_robustness_evaluation",
    "print_robustness_result",
    "save_robustness_result",
    "RobustnessResult",
    "AdversarialMetrics",
]