from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

# Allow execution with:
# python ai_pipeline/training/train_kaggle_classifier.py
BACKEND_ROOT = Path(__file__).resolve().parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ai_pipeline.datasets.kaggle.kaggle_dataset import (  # noqa: E402
    CLASS_NAMES,
    KaggleBrainTumorDataset,
    build_eval_transform,
    build_train_transform,
)
from ai_pipeline.models.kaggle_classifier import (  # noqa: E402
    KaggleBrainTumorClassifier,
    count_trainable_parameters,
)


DEFAULT_DATA_DIR = BACKEND_ROOT.parent / "datasets" / "Kaggle_data"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "ai_pipeline" / "weights"

NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_WORKERS = 0


def set_seed(seed: int) -> None:
    """
    Configure random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is preferred for the baseline.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_stratified_split(
    dataset: KaggleBrainTumorDataset,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Create a deterministic class-stratified train/validation split.

    The split is performed only on the Kaggle Training directory.

    Returns:
        train_indices, validation_indices
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    targets = np.asarray(
        [target for _, target in dataset.samples],
        dtype=np.int64,
    )

    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    validation_indices: list[int] = []

    for class_index in range(NUM_CLASSES):
        class_indices = np.flatnonzero(
            targets == class_index
        )

        if len(class_indices) < 2:
            raise ValueError(
                f"Class '{CLASS_NAMES[class_index]}' has fewer "
                "than two samples."
            )

        shuffled = class_indices.copy()
        rng.shuffle(shuffled)

        validation_count = int(
            round(len(shuffled) * validation_fraction)
        )

        validation_count = max(
            1,
            min(validation_count, len(shuffled) - 1),
        )

        validation_indices.extend(
            shuffled[:validation_count].tolist()
        )

        train_indices.extend(
            shuffled[validation_count:].tolist()
        )

    train_indices.sort()
    validation_indices.sort()

    return train_indices, validation_indices


def create_datasets(
    data_dir: Path,
    validation_fraction: float,
    seed: int,
) -> tuple[
    Subset[tuple[torch.Tensor, int, dict[str, Any]]],
    Subset[tuple[torch.Tensor, int, dict[str, Any]]],
    KaggleBrainTumorDataset,
]:
    """
    Create training, validation, and untouched testing datasets.

    Training and validation are split only from the Kaggle Training
    directory. The Kaggle Testing directory remains untouched.
    """

    training_base = KaggleBrainTumorDataset(
        root=data_dir,
        split="Training",
        transform=build_eval_transform(),
    )

    train_indices, validation_indices = create_stratified_split(
        dataset=training_base,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    # Separate dataset instances ensure training augmentation cannot
    # accidentally leak into validation.
    training_augmented = KaggleBrainTumorDataset(
        root=data_dir,
        split="Training",
        transform=build_train_transform(),
    )

    validation_deterministic = KaggleBrainTumorDataset(
        root=data_dir,
        split="Training",
        transform=build_eval_transform(),
    )

    test_dataset = KaggleBrainTumorDataset(
        root=data_dir,
        split="Testing",
        transform=build_eval_transform(),
    )

    train_dataset = Subset(
        training_augmented,
        train_indices,
    )

    validation_dataset = Subset(
        validation_deterministic,
        validation_indices,
    )

    return train_dataset, validation_dataset, test_dataset


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """
    Calculate multiclass accuracy and macro-averaged precision,
    recall, and F1 without external metric dependencies.
    """

    predictions = predictions.cpu()
    targets = targets.cpu()

    accuracy = float(
        (predictions == targets).float().mean().item()
    )

    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    for class_index in range(NUM_CLASSES):
        true_positive = int(
            ((predictions == class_index) & (targets == class_index))
            .sum()
            .item()
        )

        false_positive = int(
            ((predictions == class_index) & (targets != class_index))
            .sum()
            .item()
        )

        false_negative = int(
            ((predictions != class_index) & (targets == class_index))
            .sum()
            .item()
        )

        if true_positive + false_positive > 0:
            precision = true_positive / (
                true_positive + false_positive
            )
        else:
            precision = 0.0

        if true_positive + false_negative > 0:
            recall = true_positive / (
                true_positive + false_negative
            )
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = (
                2.0 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(
            np.mean(precision_values)
        ),
        "macro_recall": float(
            np.mean(recall_values)
        ),
        "macro_f1": float(
            np.mean(f1_values)
        ),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """
    Run one training or validation epoch.
    """

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

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

        total_loss += (
            float(loss.item()) * images.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_predictions.append(
            predictions.detach().cpu()
        )

        all_targets.append(
            targets.detach().cpu()
        )

    predictions_tensor = torch.cat(all_predictions)
    targets_tensor = torch.cat(all_targets)

    metrics = calculate_metrics(
        predictions=predictions_tensor,
        targets=targets_tensor,
    )

    metrics["loss"] = (
        total_loss / len(loader.dataset)
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the NeuroScan-AI Kaggle 2D "
            "brain tumor classification baseline."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning_rate must be > 0."
        )

    set_seed(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Kaggle data directory does not exist: {data_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== NEUROSCAN-AI KAGGLE BASELINE ===")
    print("Data directory:", data_dir)
    print("Output directory:", output_dir)
    print("Seed:", args.seed)
    print("Validation fraction:", args.validation_fraction)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    train_dataset, validation_dataset, test_dataset = (
        create_datasets(
            data_dir=data_dir,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    )

    print("\n=== DATA SPLIT ===")
    print("Train:", len(train_dataset))
    print("Validation:", len(validation_dataset))
    print("Final test:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = KaggleBrainTumorClassifier().to(device)

    parameter_count = count_trainable_parameters(model)

    print("Trainable parameters:", parameter_count)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    best_validation_f1 = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        print(
            f"\n=== EPOCH {epoch}/{args.epochs} ==="
        )

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
            f"accuracy={train_metrics['accuracy']:.6f} "
            f"macro_f1={train_metrics['macro_f1']:.6f}"
        )

        print(
            "Validation "
            f"loss={validation_metrics['loss']:.6f} "
            f"accuracy={validation_metrics['accuracy']:.6f} "
            f"macro_f1={validation_metrics['macro_f1']:.6f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )

        if (
            validation_metrics["macro_f1"]
            > best_validation_f1
        ):
            best_validation_f1 = (
                validation_metrics["macro_f1"]
            )
            best_epoch = epoch

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "seed": args.seed,
                "image_size": [224, 224],
                "input_channels": 1,
                "num_classes": NUM_CLASSES,
                "class_names": list(CLASS_NAMES),
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "validation_fraction": (
                    args.validation_fraction
                ),
                "train_samples": len(train_dataset),
                "validation_samples": len(
                    validation_dataset
                ),
                "parameter_count": parameter_count,
                "validation_metrics": validation_metrics,
            }

            torch.save(
                checkpoint,
                output_dir / "kaggle_classifier_best.pth",
            )

    experiment = {
        "seed": args.seed,
        "device": str(device),
        "data_dir": str(data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": (
            args.validation_fraction
        ),
        "train_samples": len(train_dataset),
        "validation_samples": len(
            validation_dataset
        ),
        "test_samples": len(test_dataset),
        "class_names": list(CLASS_NAMES),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": (
            best_validation_f1
        ),
        "history": history,
    }

    with open(
        output_dir / "kaggle_classifier_experiment.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment,
            file,
            indent=2,
        )

    print("\n=== TRAINING COMPLETE ===")
    print("Best epoch:", best_epoch)
    print(
        "Best validation macro F1:",
        f"{best_validation_f1:.6f}",
    )
    print(
        "Checkpoint:",
        output_dir / "kaggle_classifier_best.pth",
    )
    print(
        "Experiment:",
        output_dir / "kaggle_classifier_experiment.json",
    )


if __name__ == "__main__":
    main()