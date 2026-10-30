"""Standalone BraTS HGG/LGG cohort classification training script.

Trains a separate four-channel classifier on BraTS 2020 training subjects
with HGG/LGG dataset-cohort labels from name_mapping.csv. ARMT-GAN and the
Kaggle classifier are not used or modified.

Prototype contract:
    BraTS patient directory
        -> multimodal MRI loading (4 modalities)
        -> patient-level stratified split (70/15/15, seed 42)
        -> deterministic volume-middle-slice preprocessing
        -> HGG/LGG cohort labels from name_mapping.csv
        -> four-channel classifier (2 logits: HGG, LGG)
        -> class-weighted CrossEntropyLoss
        -> validation balanced accuracy model selection
        -> reproducible classifier checkpoint

This file intentionally does NOT implement:
    - clinical diagnosis
    - WHO grading
    - cancer staging
    - ARMT-GAN segmentation
    - Kaggle tumour-type classification
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
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# Allow execution with: python ai_pipeline/training/train_hgg_lgg_classifier.py
BACKEND_ROOT = Path(__file__).resolve().parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ai_pipeline.datasets.brats_classification import (  # noqa: E402
    CLASS_NAMES,
    CLASS_TO_INDEX,
    BraTSHggLggClassificationDataset,
    ClassificationSplitConfig,
    discover_classification_patients,
    load_brats2020_cohort_labels,
    stratified_patient_split,
)
from ai_pipeline.models.hgg_lgg_classifier import (  # noqa: E402
    HggLggClassifier,
    count_trainable_parameters,
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
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_WORKERS = 0

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
class TrainingConfig:
    """Immutable training configuration for HGG/LGG classifier reproducibility."""

    # Data
    image_size: int = 224
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 16
    num_workers: int = 0

    # Model
    in_channels: int = 4
    num_classes: int = 2

    # Optimization
    epochs: int = 10
    learning_rate: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)

    # Reproducibility
    seed: int = DEFAULT_SEED
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "adam_betas": list(self.adam_betas),
            "seed": self.seed,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
        }


BASELINE_CONFIG = TrainingConfig()


def compute_class_weights(
    train_patients: list[dict[str, Any]],
) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Does not discard LGG subjects; instead up-weights the minority class.
    """
    counts = {0: 0, 1: 0}
    for patient in train_patients:
        counts[int(patient["class_index"])] += 1

    total = sum(counts.values())
    weights: list[float] = []
    for class_index in range(NUM_CLASSES):
        count = counts[class_index]
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(total / (NUM_CLASSES * count))
    return torch.tensor(weights, dtype=torch.float32)


def balanced_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> float:
    """Balanced accuracy = mean of per-class recall."""
    predictions = predictions.cpu()
    targets = targets.cpu()
    recalls: list[float] = []
    for class_index in range(num_classes):
        tp = int(
            ((predictions == class_index) & (targets == class_index))
            .sum()
            .item()
        )
        fn = int(
            ((predictions != class_index) & (targets == class_index))
            .sum()
            .item()
        )
        denom = tp + fn
        recalls.append(tp / denom if denom > 0 else 0.0)
    return float(sum(recalls) / len(recalls))


def lgg_sensitivity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Recall for the LGG class (class index 1)."""
    predictions = predictions.cpu()
    targets = targets.cpu()
    tp = int(((predictions == 1) & (targets == 1)).sum().item())
    fn = int(((predictions != 1) & (targets == 1)).sum().item())
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def hgg_sensitivity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Recall for the HGG class (class index 0)."""
    predictions = predictions.cpu()
    targets = targets.cpu()
    tp = int(((predictions == 0) & (targets == 0)).sum().item())
    fn = int(((predictions != 0) & (targets == 0)).sum().item())
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> list[list[int]]:
    """Confusion matrix as list of lists [true_class][predicted_class]."""
    predictions = predictions.cpu()
    targets = targets.cpu()
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_index, pred_index in zip(targets.tolist(), predictions.tolist()):
        matrix[true_index][pred_index] += 1
    return matrix


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    """Run one training or validation epoch."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for images, targets, _metadata in loader:
        images = images.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        all_predictions.append(logits.argmax(dim=1).detach().cpu())
        all_targets.append(targets.detach().cpu())

    predictions_tensor = torch.cat(all_predictions)
    targets_tensor = torch.cat(all_targets)

    return {
        "loss": total_loss / len(loader.dataset),
        "balanced_accuracy": balanced_accuracy(
            predictions_tensor, targets_tensor
        ),
        "lgg_sensitivity": lgg_sensitivity(predictions_tensor, targets_tensor),
        "hgg_sensitivity": hgg_sensitivity(predictions_tensor, targets_tensor),
        "confusion_matrix": confusion_matrix(
            predictions_tensor, targets_tensor
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the NeuroScan-AI BraTS HGG/LGG cohort classifier."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be > 0.")

    set_seed(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"BraTS data directory does not exist: {data_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = data_dir / "name_mapping.csv"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"BraTS label mapping not found: {mapping_path}"
        )

    print("=== NEUROSCAN-AI HGG/LGG COHORT CLASSIFIER TRAINING ===")
    print("Data directory:", data_dir)
    print("Output directory:", output_dir)
    print("Seed:", args.seed)
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

    # Load labels and discover patients.
    labels = load_brats2020_cohort_labels(mapping_path)
    print("Labelled subjects in mapping:", len(labels))

    patients = discover_classification_patients(data_dir, labels)
    print("Complete patients with labels:", len(patients))

    # Patient-disjoint stratified split (before any slice extraction).
    split_config = ClassificationSplitConfig(
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        seed=args.seed,
        image_size=224,
    )
    splits = stratified_patient_split(patients, config=split_config)

    train_patients = splits["train"]
    validation_patients = splits["validation"]
    test_patients = splits["test"]

    print("\n=== DATA SPLIT ===")
    print("Train:", len(train_patients))
    print("Validation:", len(validation_patients))
    print("Test (held-out):", len(test_patients))

    # Verify no patient leakage across partitions.
    train_ids = {p["subject_id"] for p in train_patients}
    val_ids = {p["subject_id"] for p in validation_patients}
    test_ids = {p["subject_id"] for p in test_patients}
    assert not (train_ids & val_ids), "Train/validation leakage"
    assert not (train_ids & test_ids), "Train/test leakage"
    assert not (val_ids & test_ids), "Validation/test leakage"
    assert len(train_ids | val_ids | test_ids) == len(
        patients
    ), "Not all patients assigned"

    # Class distribution per partition.
    def grade_counts(partition: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "HGG": sum(1 for p in partition if p["grade"] == "HGG"),
            "LGG": sum(1 for p in partition if p["grade"] == "LGG"),
        }

    print("Train grades:", grade_counts(train_patients))
    print("Validation grades:", grade_counts(validation_patients))
    print("Test grades:", grade_counts(test_patients))

    # Create dataset instances (separate instances prevent augmentation leakage).
    config = PreprocessingConfig(image_size=224)
    train_dataset = BraTSHggLggClassificationDataset(train_patients, config=config)
    validation_dataset = BraTSHggLggClassificationDataset(
        validation_patients, config=config
    )
    test_dataset = BraTSHggLggClassificationDataset(test_patients, config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    model = HggLggClassifier().to(device)
    parameter_count = count_trainable_parameters(model)
    print("Trainable parameters:", parameter_count)

    class_weights = compute_class_weights(train_patients).to(device)
    print("Class weights (HGG, LGG):", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=TrainingConfig().adam_betas,
    )

    best_validation_balanced_accuracy = -1.0
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== EPOCH {epoch}/{args.epochs} ===")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        print(
            "Train     "
            f"loss={train_metrics['loss']:.6f} "
            f"balanced_acc={train_metrics['balanced_accuracy']:.6f} "
            f"lgg_sens={train_metrics['lgg_sensitivity']:.6f}"
        )
        print(
            "Validation "
            f"loss={validation_metrics['loss']:.6f} "
            f"balanced_acc={validation_metrics['balanced_accuracy']:.6f} "
            f"lgg_sens={validation_metrics['lgg_sensitivity']:.6f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )

        # Model selection by best validation balanced accuracy.
        if validation_metrics["balanced_accuracy"] > best_validation_balanced_accuracy:
            best_validation_balanced_accuracy = (
                validation_metrics["balanced_accuracy"]
            )
            best_epoch = epoch
            best_model_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

    # Load best model state before reporting.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": best_epoch,
        "seed": args.seed,
        "image_size": [224, 224],
        "input_channels": 4,
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "class_to_index": CLASS_TO_INDEX,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "train_fraction": split_config.train_fraction,
        "validation_fraction": split_config.validation_fraction,
        "test_fraction": split_config.test_fraction,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "parameter_count": parameter_count,
        "best_validation_balanced_accuracy": best_validation_balanced_accuracy,
        "train_patient_ids": sorted(train_ids),
        "validation_patient_ids": sorted(val_ids),
        "test_patient_ids": sorted(test_ids),
        "training_config": config.to_dict(),
    }

    torch.save(checkpoint, output_dir / "hgg_lgg_classifier_best.pth")

    experiment = {
        "seed": args.seed,
        "device": str(device),
        "data_dir": str(data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_fraction": split_config.train_fraction,
        "validation_fraction": split_config.validation_fraction,
        "test_fraction": split_config.test_fraction,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "class_names": list(CLASS_NAMES),
        "class_to_index": CLASS_TO_INDEX,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_validation_balanced_accuracy,
        "history": history,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(
        output_dir / "hgg_lgg_classifier_experiment.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(experiment, f, indent=2)

    print("\n=== TRAINING COMPLETE ===")
    print("Best epoch:", best_epoch)
    print(
        "Best validation balanced accuracy:",
        f"{best_validation_balanced_accuracy:.6f}",
    )
    print("Checkpoint:", output_dir / "hgg_lgg_classifier_best.pth")
    print("Experiment:", output_dir / "hgg_lgg_classifier_experiment.json")


if __name__ == "__main__":
    main()
