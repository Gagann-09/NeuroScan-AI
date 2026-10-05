"""
NeuroScan AI - Robustness Evaluation Pipeline

Integrates FGSM and PGD adversarial evaluation with the frozen P3 evaluation protocol.
Reuses P3 evaluation machinery (metrics, patient splitting, artifact preservation).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from ai_pipeline.evaluation.evaluate_armt_gan_baseline import (
    EvaluationProtocol,
    PatientRecord,
    MetricSummary,
    load_checkpoint,
    discover_complete_patients,
    split_patients_three_way,
    load_patient_slices,
    dice_score,
    iou_score,
    precision_score,
    sensitivity_score,
    summarize,
    BASELINE_PROTOCOL,
    set_seed,
)
from ai_pipeline.models.armt_gan import ARMTGenerator2D
from ai_pipeline.robustness.attacks import (
    AttackConfig,
    soft_dice_loss,
    fgsm_attack,
    pgd_attack,
    compute_perturbation_norm,
    verify_attack_constraints,
    BASELINE_ATTACK_CONFIG,
)
from ai_pipeline.robustness.config import RobustnessConfig


@dataclass
class AdversarialMetrics:
    """Metrics for a single adversarial evaluation."""
    attack_type: str          # "clean", "fgsm", "pgd"
    epsilon: float
    step_size: Optional[float]
    num_iterations: Optional[int]
    seed: int
    
    # Slice-level metrics
    dice: MetricSummary
    iou: MetricSummary
    precision: MetricSummary
    sensitivity: MetricSummary
    
    # Patient-level metrics
    patient_dice: MetricSummary
    patient_iou: MetricSummary
    patient_precision: MetricSummary
    patient_sensitivity: MetricSummary
    
    # Sample counts
    evaluated_slices: int
    skipped_slices: int
    evaluated_patients: int
    
    # Latency
    total_inference_seconds: float
    mean_slice_latency_ms: float
    
    # Clean baseline metrics (for delta computation)
    clean_dice_mean: Optional[float] = None
    clean_iou_mean: Optional[float] = None
    clean_precision_mean: Optional[float] = None
    clean_sensitivity_mean: Optional[float] = None


@dataclass
class RobustnessResult:
    """Complete robustness evaluation result."""
    checkpoint: str
    device: str
    protocol: dict  # EvaluationProtocol serialized
    attack_config: dict  # AttackConfig serialized
    
    clean_metrics: AdversarialMetrics
    fgsm_metrics: Optional[AdversarialMetrics] = None
    pgd_metrics: Optional[AdversarialMetrics] = None
    
    # Summary deltas
    fgsm_dice_delta: Optional[float] = None
    pgd_dice_delta: Optional[float] = None
    fgsm_iou_delta: Optional[float] = None
    pgd_iou_delta: Optional[float] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_clean(
    model: nn.Module,
    patients: list[PatientRecord],
    device: torch.device,
    protocol: EvaluationProtocol,
    output_dir: Optional[Path] = None,
) -> tuple[
    list[float], list[float], list[float], list[float],
    dict[str, dict[str, float]],
    int, int, float, float,
    list[dict],
]:
    """
    Clean evaluation (reuses P3 evaluate function).
    Returns slice metrics, patient metrics, counts, latency, raw artifacts.
    """
    from ai_pipeline.evaluation.evaluate_armt_gan_baseline import evaluate as p3_evaluate
    return p3_evaluate(model, patients, device, protocol, output_dir)


def evaluate_adversarial(
    model: nn.Module,
    patients: list[PatientRecord],
    device: torch.device,
    protocol: EvaluationProtocol,
    attack_config: AttackConfig,
    attack_type: str,  # "fgsm" or "pgd"
    output_dir: Optional[Path] = None,
) -> tuple[
    list[float], list[float], list[float], list[float],
    dict[str, dict[str, float]],
    int, int, float, float,
    list[dict],
]:
    """
    Evaluate model on adversarially perturbed test patients.
    
    Uses the same evaluation logic as P3 but applies FGSM/PGD to each slice.
    """
    slice_dice: list[float] = []
    slice_iou: list[float] = []
    slice_precision: list[float] = []
    slice_sensitivity: list[float] = []

    patient_metrics: dict[str, dict[str, float]] = {}

    evaluated_slices = 0
    skipped_slices = 0

    total_inference_seconds = 0.0
    inference_calls = 0

    raw_artifacts: list[dict] = []

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)

    for patient in sorted(patients, key=lambda record: record.patient_id):
        samples = load_patient_slices(
            patient=patient,
            image_size=protocol.image_size,
            min_tumor_pixels=protocol.min_tumor_pixels,
        )

        if not samples:
            skipped_slices += 1
            continue

        patient_dice: list[float] = []
        patient_iou: list[float] = []
        patient_precision: list[float] = []
        patient_sensitivity: list[float] = []

        for image_np, mask_np, slice_index in samples:
            image = torch.from_numpy(image_np).unsqueeze(0).to(device)  # [1, 4, H, W]
            target = torch.from_numpy(mask_np).to(device)  # [1, H, W]

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            # Apply adversarial attack
            if attack_type == "fgsm":
                from ai_pipeline.robustness.attacks import fgsm_attack
                adv_image = fgsm_attack(
                    model=model,
                    images=image,
                    target_masks=target.unsqueeze(1).float(),  # [1, 1, H, W]
                    config=attack_config,
                )
            elif attack_type == "pgd":
                from ai_pipeline.robustness.attacks import pgd_attack
                adv_image = pgd_attack(
                    model=model,
                    images=image,
                    target_masks=target.unsqueeze(1).float(),
                    config=attack_config,
                )
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            # Verify constraints
            from ai_pipeline.robustness.attacks import verify_attack_constraints
            passed, msg = verify_attack_constraints(adv_image, image, BASELINE_ATTACK_CONFIG)
            if not passed:
                raise RuntimeError(f"Attack constraint violation: {msg}")

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.inference_mode():
                prediction = model(adv_image)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            total_inference_seconds += elapsed
            inference_calls += 1

            # Compute metrics (same as P3)
            pred_prob = prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
            pred_binary = pred_prob >= protocol.threshold
            target_binary = (mask_np.squeeze(0) >= 0.5)

            dice = dice_score(pred_binary, target_binary)
            iou = iou_score(pred_binary, target_binary)
            precision = precision_score(pred_binary, target_binary)
            sensitivity = sensitivity_score(pred_binary, target_binary)

            slice_dice.append(dice)
            slice_iou.append(iou)
            slice_precision.append(precision)
            slice_sensitivity.append(sensitivity)

            patient_dice.append(dice)
            patient_iou.append(iou)
            patient_precision.append(precision)
            patient_sensitivity.append(sensitivity)

            evaluated_slices += 1

            # Save raw adversarial prediction artifact
            if output_dir is not None:
                pred_filename = f"{patient.patient_id}_slice{slice_index:03d}_{attack_type}_eps{attack_config.epsilon:.4f}_pred.npy"
                pred_path = pred_dir / pred_filename
                np.save(pred_path, pred_prob.astype(np.float32))

                raw_artifacts.append({
                    "patient_id": patient.patient_id,
                    "slice_index": slice_index,
                    "attack_type": attack_type,
                    "epsilon": attack_config.epsilon,
                    "prediction_path": str(pred_path.relative_to(output_dir)),
                    "shape": list(pred_prob.shape),
                    "min": float(pred_prob.min()),
                    "max": float(pred_prob.max()),
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "sensitivity": sensitivity,
                })

        patient_metrics[patient.patient_id] = {
            "dice": float(mean(patient_dice)),
            "iou": float(mean(patient_iou)),
            "precision": float(mean(patient_precision)),
            "sensitivity": float(mean(patient_sensitivity)),
            "evaluated_slices": len(samples),
        }

    mean_batch_latency_ms = (
        total_inference_seconds / inference_calls * 1000.0
        if inference_calls
        else 0.0
    )

    mean_slice_latency_ms = mean_batch_latency_ms

    return (
        slice_dice,
        slice_iou,
        slice_precision,
        slice_sensitivity,
        patient_metrics,
        evaluated_slices,
        skipped_slices,
        total_inference_seconds,
        mean_slice_latency_ms,
        raw_artifacts,
    )


def build_adversarial_metrics(
    attack_type: str,
    attack_config: AttackConfig,
    protocol: EvaluationProtocol,
    slice_dice: list[float],
    slice_iou: list[float],
    slice_precision: list[float],
    slice_sensitivity: list[float],
    patient_metrics: dict[str, dict[str, float]],
    evaluated_slices: int,
    skipped_slices: int,
    total_inference_seconds: float,
    mean_slice_latency_ms: float,
) -> AdversarialMetrics:
    """Build AdversarialMetrics from evaluation results."""
    patient_dice = [m["dice"] for m in patient_metrics.values()]
    patient_iou = [m["iou"] for m in patient_metrics.values()]
    patient_precision = [m["precision"] for m in patient_metrics.values()]
    patient_sensitivity = [m["sensitivity"] for m in patient_metrics.values()]

    return AdversarialMetrics(
        attack_type=attack_type,
        epsilon=attack_config.epsilon,
        step_size=attack_config.step_size if attack_type == "pgd" else None,
        num_iterations=attack_config.num_iterations if attack_type == "pgd" else None,
        seed=attack_config.seed,
        dice=summarize(slice_dice),
        iou=summarize(slice_iou),
        precision=summarize(slice_precision),
        sensitivity=summarize(slice_sensitivity),
        patient_dice=summarize(patient_dice),
        patient_iou=summarize(patient_iou),
        patient_precision=summarize(patient_precision),
        patient_sensitivity=summarize(patient_sensitivity),
        evaluated_slices=evaluated_slices,
        skipped_slices=skipped_slices,
        evaluated_patients=len(patient_metrics),
        total_inference_seconds=total_inference_seconds,
        mean_slice_latency_ms=mean_slice_latency_ms,
    )


def run_robustness_evaluation(
    config: RobustnessConfig,
) -> RobustnessResult:
    """
    Main entry point for robustness evaluation.
    
    Runs clean, FGSM, and PGD evaluation on the test split.
    """
    protocol = config.protocol
    attack_config = config.attack_config

    set_seed(protocol.seed)

    # Resolve device
    if protocol.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available.")

    device = torch.device(
        "cuda" if (protocol.device == "cuda" or (protocol.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    output_dir = Path(config.output_dir) if config.output_dir else None

    print("Discovering complete labeled BraTS patients...")
    all_patients = discover_complete_patients(Path(config.data_dir))

    training_patients, validation_patients, test_patients = split_patients_three_way(
        patients=all_patients,
        train_fraction=protocol.train_fraction,
        validation_fraction=protocol.validation_fraction,
        test_fraction=protocol.test_fraction,
        seed=protocol.seed,
    )

    print(f"Complete labeled patients: {len(all_patients)}")
    print(f"Training split:            {len(training_patients)}")
    print(f"Validation split:          {len(validation_patients)}")
    print(f"Test split:                {len(test_patients)}")

    model, checkpoint_metadata = load_checkpoint(
        checkpoint_path=Path(config.checkpoint),
        device=device,
    )

    if checkpoint_metadata["seed"] != protocol.seed:
        raise RuntimeError(
            f"Checkpoint seed does not match evaluator seed: "
            f"{checkpoint_metadata['seed']} != {protocol.seed}"
        )

    if checkpoint_metadata["image_size"] != protocol.image_size:
        raise RuntimeError(
            f"Checkpoint image size does not match evaluator config: "
            f"{checkpoint_metadata['image_size']} != {protocol.image_size}"
        )

    # ---- Clean Evaluation ----
    print("\n" + "=" * 60)
    print("CLEAN BASELINE EVALUATION")
    print("=" * 60)

    clean_output_dir = output_dir / "clean" if output_dir else None

    (
        clean_slice_dice,
        clean_slice_iou,
        clean_slice_precision,
        clean_slice_sensitivity,
        clean_patient_metrics,
        clean_eval_slices,
        clean_skipped_slices,
        clean_total_time,
        clean_latency,
        clean_raw_artifacts,
    ) = evaluate_clean(
        model=model,
        patients=test_patients,
        device=device,
        protocol=protocol,
        output_dir=clean_output_dir,
    )

    clean_metrics = build_adversarial_metrics(
        attack_type="clean",
        attack_config=AttackConfig(epsilon=0.0),  # Clean has no epsilon
        protocol=protocol,
        slice_dice=clean_slice_dice,
        slice_iou=clean_slice_iou,
        slice_precision=clean_slice_precision,
        slice_sensitivity=clean_slice_sensitivity,
        patient_metrics=clean_patient_metrics,
        evaluated_slices=clean_eval_slices,
        skipped_slices=clean_skipped_slices,
        total_inference_seconds=clean_total_time,
        mean_slice_latency_ms=clean_latency,
    )

    print(f"Clean Dice:   {clean_metrics.dice.mean:.6f}")
    print(f"Clean IoU:    {clean_metrics.iou.mean:.6f}")
    print(f"Clean Prec:   {clean_metrics.precision.mean:.6f}")
    print(f"Clean Sens:   {clean_metrics.sensitivity.mean:.6f}")

    # ---- FGSM Evaluation ----
    print("\n" + "=" * 60)
    print(f"FGSM EVALUATION (eps={BASELINE_ATTACK_CONFIG.epsilon})")
    print("=" * 60)

    fgsm_output_dir = output_dir / f"fgsm_eps{BASELINE_ATTACK_CONFIG.epsilon:.4f}" if output_dir else None

    (
        fgsm_slice_dice,
        fgsm_slice_iou,
        fgsm_slice_precision,
        fgsm_slice_sensitivity,
        fgsm_patient_metrics,
        fgsm_eval_slices,
        fgsm_skipped_slices,
        fgsm_total_time,
        fgsm_latency,
        fgsm_raw_artifacts,
    ) = evaluate_adversarial(
        model=model,
        patients=test_patients,
        device=device,
        protocol=protocol,
        attack_config=BASELINE_ATTACK_CONFIG,
        attack_type="fgsm",
        output_dir=fgsm_output_dir,
    )

    fgsm_metrics = build_adversarial_metrics(
        attack_type="fgsm",
        attack_config=BASELINE_ATTACK_CONFIG,
        protocol=protocol,
        slice_dice=fgsm_slice_dice,
        slice_iou=fgsm_slice_iou,
        slice_precision=fgsm_slice_precision,
        slice_sensitivity=fgsm_slice_sensitivity,
        patient_metrics=fgsm_patient_metrics,
        evaluated_slices=fgsm_eval_slices,
        skipped_slices=fgsm_skipped_slices,
        total_inference_seconds=fgsm_total_time,
        mean_slice_latency_ms=fgsm_latency,
    )

    fgsm_metrics.clean_dice_mean = clean_metrics.dice.mean
    fgsm_metrics.clean_iou_mean = clean_metrics.iou.mean
    fgsm_metrics.clean_precision_mean = clean_metrics.precision.mean
    fgsm_metrics.clean_sensitivity_mean = clean_metrics.sensitivity.mean

    print(f"FGSM Dice:   {fgsm_metrics.dice.mean:.6f} (d={fgsm_metrics.dice.mean - clean_metrics.dice.mean:+.6f})")
    print(f"FGSM IoU:    {fgsm_metrics.iou.mean:.6f} (d={fgsm_metrics.iou.mean - clean_metrics.iou.mean:+.6f})")
    print(f"FGSM Prec:   {fgsm_metrics.precision.mean:.6f} (d={fgsm_metrics.precision.mean - clean_metrics.precision.mean:+.6f})")
    print(f"FGSM Sens:   {fgsm_metrics.sensitivity.mean:.6f} (d={fgsm_metrics.sensitivity.mean - clean_metrics.sensitivity.mean:+.6f})")

    # ---- PGD Evaluation ----
    print("\n" + "=" * 60)
    print(f"PGD EVALUATION (eps={BASELINE_ATTACK_CONFIG.epsilon}, alpha={BASELINE_ATTACK_CONFIG.step_size}, K={BASELINE_ATTACK_CONFIG.num_iterations})")
    print("=" * 60)

    pgd_output_dir = output_dir / f"pgd_eps{BASELINE_ATTACK_CONFIG.epsilon:.4f}_alpha{BASELINE_ATTACK_CONFIG.step_size:.4f}_iter{BASELINE_ATTACK_CONFIG.num_iterations}" if output_dir else None

    (
        pgd_slice_dice,
        pgd_slice_iou,
        pgd_slice_precision,
        pgd_slice_sensitivity,
        pgd_patient_metrics,
        pgd_eval_slices,
        pgd_skipped_slices,
        pgd_total_time,
        pgd_latency,
        pgd_raw_artifacts,
    ) = evaluate_adversarial(
        model=model,
        patients=test_patients,
        device=device,
        protocol=protocol,
        attack_config=BASELINE_ATTACK_CONFIG,
        attack_type="pgd",
        output_dir=pgd_output_dir,
    )

    pgd_metrics = build_adversarial_metrics(
        attack_type="pgd",
        attack_config=BASELINE_ATTACK_CONFIG,
        protocol=protocol,
        slice_dice=pgd_slice_dice,
        slice_iou=pgd_slice_iou,
        slice_precision=pgd_slice_precision,
        slice_sensitivity=pgd_slice_sensitivity,
        patient_metrics=pgd_patient_metrics,
        evaluated_slices=pgd_eval_slices,
        skipped_slices=pgd_skipped_slices,
        total_inference_seconds=pgd_total_time,
        mean_slice_latency_ms=pgd_latency,
    )

    pgd_metrics.clean_dice_mean = clean_metrics.dice.mean
    pgd_metrics.clean_iou_mean = clean_metrics.iou.mean
    pgd_metrics.clean_precision_mean = clean_metrics.precision.mean
    pgd_metrics.clean_sensitivity_mean = clean_metrics.sensitivity.mean

    print(f"PGD Dice:   {pgd_metrics.dice.mean:.6f} (d={pgd_metrics.dice.mean - clean_metrics.dice.mean:+.6f})")
    print(f"PGD IoU:    {pgd_metrics.iou.mean:.6f} (d={pgd_metrics.iou.mean - clean_metrics.iou.mean:+.6f})")
    print(f"PGD Prec:   {pgd_metrics.precision.mean:.6f} (d={pgd_metrics.precision.mean - clean_metrics.precision.mean:+.6f})")
    print(f"PGD Sens:   {pgd_metrics.sensitivity.mean:.6f} (d={pgd_metrics.sensitivity.mean - clean_metrics.sensitivity.mean:+.6f})")

    # ---- Build Result ----
    fgsm_dice_delta = fgsm_metrics.dice.mean - clean_metrics.dice.mean
    pgd_dice_delta = pgd_metrics.dice.mean - clean_metrics.dice.mean
    fgsm_iou_delta = fgsm_metrics.iou.mean - clean_metrics.iou.mean
    pgd_iou_delta = pgd_metrics.iou.mean - clean_metrics.iou.mean

    result = RobustnessResult(
        checkpoint=config.checkpoint,
        device=str(device),
        protocol=protocol.to_dict(),
        attack_config=BASELINE_ATTACK_CONFIG.to_dict(),
        clean_metrics=clean_metrics,
        fgsm_metrics=fgsm_metrics,
        pgd_metrics=pgd_metrics,
        fgsm_dice_delta=fgsm_dice_delta,
        pgd_dice_delta=pgd_dice_delta,
        fgsm_iou_delta=fgsm_iou_delta,
        pgd_iou_delta=pgd_iou_delta,
    )

    return result


def print_robustness_result(result: RobustnessResult) -> None:
    """Print formatted robustness evaluation result."""
    print("\n" + "=" * 72)
    print("ARMT-GAN ROBUSTNESS EVALUATION")
    print("=" * 72)

    print(f"\nCheckpoint:             {result.checkpoint}")
    print(f"Device:                 {result.device}")
    print(f"Seed:                   {result.protocol['seed']}")
    print(f"Image size:             {result.protocol['image_size']}x{result.protocol['image_size']}")

    print("\nCLEAN BASELINE")
    c = result.clean_metrics
    print(f"  Dice:        {c.dice.mean:.6f}")
    print(f"  IoU:         {c.iou.mean:.6f}")
    print(f"  Precision:   {c.precision.mean:.6f}")
    print(f"  Sensitivity: {c.sensitivity.mean:.6f}")

    if result.fgsm_metrics:
        f = result.fgsm_metrics
        print(f"\nFGSM (ε={f.epsilon})")
        print(f"  Dice:        {f.dice.mean:.6f}  (Δ={f.dice.mean - f.clean_dice_mean:+.6f})")
        print(f"  IoU:         {f.iou.mean:.6f}  (Δ={f.iou.mean - f.clean_iou_mean:+.6f})")
        print(f"  Precision:   {f.precision.mean:.6f}  (Δ={f.precision.mean - f.clean_precision_mean:+.6f})")
        print(f"  Sensitivity: {f.sensitivity.mean:.6f}  (Δ={f.sensitivity.mean - f.clean_sensitivity_mean:+.6f})")

    if result.pgd_metrics:
        p = result.pgd_metrics
        print(f"\nPGD (ε={p.epsilon}, α={p.step_size}, K={p.num_iterations})")
        print(f"  Dice:        {p.dice.mean:.6f}  (Δ={p.dice.mean - p.clean_dice_mean:+.6f})")
        print(f"  IoU:         {p.iou.mean:.6f}  (Δ={p.iou.mean - p.clean_iou_mean:+.6f})")
        print(f"  Precision:   {p.precision.mean:.6f}  (Δ={p.precision.mean - p.clean_precision_mean:+.6f})")
        print(f"  Sensitivity: {p.sensitivity.mean:.6f}  (Δ={p.sensitivity.mean - p.clean_sensitivity_mean:+.6f})")

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("This is a held-out test evaluation on synthetic data.")
    print("No clinical diagnostic or clinical-validation claim is implied.")
    print("Real BraTS robustness evaluation pending data availability.")
    print("=" * 72)


def save_robustness_result(result: RobustnessResult, output_json: Path) -> None:
    """Save robustness result as JSON experiment record."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nJSON experiment record written to: {output_json}")