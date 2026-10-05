"""
NeuroScan AI - Robustness Attack Configuration

Frozen attack configuration for FGSM and PGD evaluation.
All parameters are immutable and versioned.
"""

from __future__ import annotations
from dataclasses import dataclass

from ai_pipeline.evaluation.evaluate_armt_gan_baseline import EvaluationProtocol
from ai_pipeline.robustness.attacks import AttackConfig, BASELINE_ATTACK_CONFIG


@dataclass(frozen=True)
class RobustnessConfig:
    """
    Immutable robustness evaluation configuration.
    """
    protocol: "EvaluationProtocol"  # Forward reference
    attack_config: AttackConfig = BASELINE_ATTACK_CONFIG
    data_dir: str = ""
    checkpoint: str = ""
    output_dir: str = ""
    save_predictions: bool = True
    
    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol.to_dict(),
            "attack_config": self.attack_config.to_dict(),
            "data_dir": self.data_dir,
            "checkpoint": self.checkpoint,
            "output_dir": self.output_dir,
            "save_predictions": self.save_predictions,
        }


BASELINE_ROBUSTNESS_CONFIG = RobustnessConfig(
    protocol=None,  # Will be set at runtime
    attack_config=BASELINE_ATTACK_CONFIG,
)