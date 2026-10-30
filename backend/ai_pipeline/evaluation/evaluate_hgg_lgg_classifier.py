"""Standalone BraTS HGG/LGG cohort classification evaluation script.

Evaluates a trained classifier checkpoint on a held-out BraTS test partition.
Reports balanced accuracy, LGG sensitivity, ordinary accuracy, HGG sensitivity,
confusion matrix, class counts, and patient-level bootstrap confidence intervals.

ARMT-GAN and the Kaggle classifier are not used or modified.

This file intentionally does NOT implement:
    - clinical diagnosis
    - WHO grading
    - cancer staging
    - API, database, dashboard, or PDF integration
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow execution with: python ai_pipeline/evaluation/evaluate_hgg_lgg_classifier.py
BACKEND_ROOT = Path(__file__).resolve().parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ai_pipeline.datasets.brats_classification import (  # noqa: E402
    CLASS_NAMES,
    BraTSHggLggClassificationDataset,
    ClassificationSplitConfig,
    discover_classification_patients,
    load_brats2020_cohort_labels,
    stratified_patient_split,
)
from ai_pipeline.models.hgg_lgg_classifier import (  # noqa: E402
    HggLggClassifier,
)
from ai_pipeline.preprocessing import PreprocessingConfig  # noqa: E402


DEFAULT_DATA_DIR = (
    BACKEND_ROOT.parent
    / "datasets"
    / "BraTS_data"
    / "BraTS2020_TrainingData"
)
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "ai_pipeline" / "weights"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0
DEFAULT_BOOTSTRAP_REPLICATES = 1000
NUM_CLASSES = len(CLASS_NAMES)


def set_seed(seed: int) -> None:
    """Configure random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable evaluation configuration for reproducibility."""

    image_size: int = 224
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    seed: int = DEFAULT_SEED
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
    ci_alpha: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "bootstrap_replicates": self.bootstrap_replicates,
            "ci_alpha": self.ci_alpha,
        }


BASELINE_CONFIG = EvaluationConfig()


def balanced_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> float:
    """Balanced accuracy = mean of per-class recall."""
    recalls: list[float] = []
    for class_index in range(num_classes):
        tp = int(((predictions == class_index) & (targets == class_index)).sum())
        fn = int(((predictions != class_index) & (targets == class_index)).sum())
        denom = tp + fn
        recalls.append(tp / denom if denom > 0 else 0.0)
    return float(sum(recalls) / len(recalls))


def lgg_sensitivity(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Recall for the LGG class (class index 1)."""
    tp = int(((predictions == 1) & (targets == 1)).sum())
    fn = int(((predictions != 1) & (targets == 1)).sum())
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def hgg_sensitivity(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Recall for the HGG class (class index 0)."""
    tp = int(((predictions == 0) & (targets == 0)).sum())
    fn = int(((predictions != 0) & (targets == 0)).sum())
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def ordinary_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float((predictions == targets).mean())


def confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> list[list[int]]:
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_index, pred_index in zip(targets.tolist(), predictions.tolist()):
        matrix[true_index][pred_index] += 1
    return matrix


def majority_class_baseline(targets: np.ndarray) -> dict[str, float]:
    """Majority-class baseline (always predict HGG / class 0)."""
    predictions = np.zeros_like(targets)
    return {
        "balanced_accuracy": balanced_accuracy(predictions, targets),
        "lgg_sensitivity": lgg_sensitivity(predictions, targets),
        "ordinary_accuracy": ordinary_accuracy(predictions, targets),
        "hgg_sensitivity": hgg_sensitivity(predictions, targets),
    }


def bootstrap_ci(
    metric_fn: callable,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Patient-level class-stratified bootstrap percentile CI.

    Resamples patients (not slices) with replacement, preserving the HGG/LGG
    ratio within each resample. Returns point estimate, lower/upper CI bounds,
    and the number of valid replicates.
    """
    rng = random.Random(seed)
    n = len(targets)
    hgg_indices = np.where(targets == 0)[0]
    lgg_indices = np.where(targets == 1)[0]

    estimates: list[float] = []
    valid_replicates = 0

    for _ in range(n_replicates):
        # Resample within each class to preserve class ratio.
        resampled_hgg = [
            int(rng.choice(hgg_indices)) for _ in range(len(hgg_indices))
        ] if len(hgg_indices) > 0 else []
        resampled_lgg = [
            int(rng.choice(lgg_indices)) for _ in range(len(lgg_indices))
        ] if len(lgg_indices) > 0 else []
        resampled_indices = sorted(resampled_hgg + resampled_lgg)

        if len(resampled_indices) == 0:
            continue

        resampled_predictions = predictions[resampled_indices]
        resampled_targets = targets[resampled_indices]

        # Skip if a resample contains only one class (metric undefined).
        if len(np.unique(resampled_targets)) < 2:
            continue

        try:
            estimate = metric_fn(resampled_predictions, resampled_targets)
        except Exception:
            continue

        estimates.append(estimate)
        valid_replicates += 1

    if len(estimates) == 0:
        return {
            "lower": None,
            "upper": None,
            "valid_replicates": 0,
            "replicates_requested": n_replicates,
        }

    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    lower = float(np.percentile(estimates, lower_percentile))
    upper = float(np.percentile(estimates, upper_percentile))

    return {
        "lower": lower,
        "upper": upper,
        "valid_replicates": valid_replicates,
        "replicates_requested": n_replicates,
    }


def evaluate(
    model: HggLggClassifier,
    loader: DataLoader,
    device: torch.device,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Run evaluation on the held-out test partition."""
    model.eval()
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, targets, _metadata in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_probabilities.append(probabilities.cpu())

    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()
    probabilities = torch.cat(all_probabilities).numpy()

    ba = balanced_accuracy(predictions, targets)
    lgg_sens = lgg_sensitivity(predictions, targets)
    hgg_sens = hgg_sensitivity(predictions, targets)
    acc = ordinary_accuracy(predictions, targets)
    cm = confusion_matrix(predictions, targets)

    baseline = majority_class_baseline(targets)

    ba_ci = bootstrap_ci(balanced_accuracy, predictions, targets,
                         config.bootstrap_replicates, config.ci_alpha, config.seed)
    lgg_ci = bootstrap_ci(lgg_sensitivity, predictions, targets,
                          config.bootstrap_replicates, config.ci_alpha, config.seed)
    hgg_ci = bootstrap_ci(hgg_sensitivity, predictions, targets,
                          config.bootstrap_replicates, config.ci_alpha, config.seed)

    # ROC-AUC where estimable (requires both classes in test set).
    roc_auc = None
    roc_auc_ci = None
    if len(np.unique(targets)) == 2:
        # Simple ROC-AUC via Mann-Whitney U on LGG probabilities.
        lgg_probs = probabilities[:, 1]
        hgg_mask = targets == 0
        lgg_mask = targets == 1
        n_hgg = int(hgg_mask.sum())
        n_lgg = int(lgg_mask.sum())
        if n_hgg > 0 and n_lgg > 0:
            # Rank-based Mann-Whitney U.
            order = np.argsort(lgg_probs)
            ranks = np.empty(len(lgg_probs), dtype=float)
            ranks[order] = np.arange(1, len(lgg_probs) + 1)
            # Handle ties with average ranks.
            sorted_probs = lgg_probs[order]
            unique_probs = np.unique(sorted_probs)
            for prob in unique_probs:
                tied_mask = sorted_probs == prob
                tied_ranks = ranks[order][tied_mask]
                ranks[order][tied_mask] = tied_ranks.mean()
            sum_ranks_lgg = float(ranks[lgg_mask].sum())
            u = sum_ranks_lgg - n_lgg * (n_lgg + 1) / 2
            roc_auc = float(u / (n_hgg * n_lgg))

    return {
        "balanced_accuracy": ba,
        "balanced_accuracy_ci": ba_ci,
        "lgg_sensitivity": lgg_sens,
        "lgg_sensitivity_ci": lgg_ci,
        "hgg_sensitivity": hgg_sens,
        "hgg_sensitivity_ci": hgg_ci,
        "ordinary_accuracy": acc,
        "confusion_matrix": cm,
        "majority_class_baseline": baseline,
        "roc_auc": roc_auc,
        "test_class_counts": {
            "HGG": int((targets == 0).sum()),
            "LGG": int((targets == 1).sum()),
            "total": int(len(targets)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the NeuroScan-AI BraTS HGG/LGG cohort classifier."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--bootstrap_replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    set_seed(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = data_dir / "name_mapping.csv"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"BraTS label mapping not found: {mapping_path}"
        )

    print("=== NEUROSCAN-AI HGG/LGG COHORT CLASSIFIER EVALUATION ===")
    print("Data directory:", data_dir)
    print("Checkpoint:", args.checkpoint)
    print("Seed:", args.seed)

    # Load labels and discover patients.
    labels = load_brats2020_cohort_labels(mapping_path)
    patients = discover_classification_patients(data_dir, labels)

    # Patient-disjoint stratified split (same seed/fractions as training).
    split_config = ClassificationSplitConfig(
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=args.seed,
        image_size=224,
    )
    splits = stratified_patient_split(patients, config=split_config)
    test_patients = splits["test"]

    print("Test patients:", len(test_patients))

    # Verify test set class counts.
    test_hgg = sum(1 for p in test_patients if p["grade"] == "HGG")
    test_lgg = sum(1 for p in test_patients if p["grade"] == "LGG")
    print("Test grades - HGG:", test_hgg, "LGG:", test_lgg)

    # Create test dataset (no augmentation).
    config = PreprocessingConfig(image_size=224)
    test_dataset = BraTSHggLggClassificationDataset(test_patients, config=config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # Load checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = HggLggClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Evaluate.
    eval_config = EvaluationConfig(
        image_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    results = evaluate(model, test_loader, device, eval_config)

    print("\n=== EVALUATION RESULTS ===")
    print("Balanced accuracy:", f"{results['balanced_accuracy']:.6f}")
    ba_ci = results["balanced_accuracy_ci"]
    print(
        "  CI 95%: [",
        f"{ba_ci['lower']:.6f}" if ba_ci['lower'] is not None else "None",
        ", ",
        f"{ba_ci['upper']:.6f}" if ba_ci['upper'] is not None else "None",
        "]",
        "valid replicates:",
        f"{ba_ci['valid_replicates']}/{ba_ci['replicates_requested']}",
    )
    print("LGG sensitivity:", f"{results['lgg_sensitivity']:.6f}")
    lgg_ci = results["lgg_sensitivity_ci"]
    print(
        "  CI 95%: [",
        f"{lgg_ci['lower']:.6f}" if lgg_ci['lower'] is not None else "None",
        ", ",
        f"{lgg_ci['upper']:.6f}" if lgg_ci['upper'] is not None else "None",
        "]",
        "valid replicates:",
        f"{lgg_ci['valid_replicates']}/{lgg_ci['replicates_requested']}",
    )
    print("HGG sensitivity:", f"{results['hgg_sensitivity']:.6f}")
    print("Ordinary accuracy:", f"{results['ordinary_accuracy']:.6f}")
    print("Confusion matrix:", results["confusion_matrix"])
    print("Test class counts:", results["test_class_counts"])
    print("Majority-class baseline:", results["majority_class_baseline"])

    # Save results.
    experiment_record = {
        "seed": args.seed,
        "device": str(device),
        "data_dir": str(data_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_seed": checkpoint.get("seed"),
        "test_patient_ids": sorted(p["subject_id"] for p in test_patients),
        "test_class_counts": results["test_class_counts"],
        "results": results,
        "evaluation_config": eval_config.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(
        output_dir / "hgg_lgg_classifier_evaluation.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(experiment_record, f, indent=2)

    print("\nEvaluation record saved to:")
    print(output_dir / "hgg_lgg_classifier_evaluation.json")


if __name__ == "__main__":
    main()
