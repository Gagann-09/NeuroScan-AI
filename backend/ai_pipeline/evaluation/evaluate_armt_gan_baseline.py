"""
Standalone evaluator for the 2D ARMT-GAN BraTS prototype.

Purpose
-------
Evaluate a current-format ARMT-GAN generator checkpoint on the exact
seed-42 held-out patient split used by the prototype training pipeline.

Important
---------
- This evaluator is READ-ONLY with respect to dataset and checkpoints.
- It does not train or modify model weights.
- It evaluates only complete labeled BraTS training patients.
- The resulting metrics represent a held-out validation set, NOT an
  independent test set.
- No synthetic masks or fabricated metrics are used.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ai_pipeline.models.armt_gan import ARMTGenerator2D
from ai_pipeline.preprocessing import BraTSPreprocessor, PreprocessingConfig


MODALITY_SUFFIXES = (
    "_t1.nii",
    "_t1ce.nii",
    "_t2.nii",
    "_flair.nii",
)

MODALITY_GZ_SUFFIXES = (
    "_t1.nii.gz",
    "_t1ce.nii.gz",
    "_t2.nii.gz",
    "_flair.nii.gz",
)

MODALITY_NEW_SUFFIXES = (
    "-t1n.nii.gz",
    "-t1c.nii.gz",
    "-t2w.nii.gz",
    "-t2f.nii.gz",
)

SEGMENTATION_SUFFIXES = (
    "_seg.nii",
    "_seg.nii.gz",
    "-seg.nii.gz",
)

DEFAULT_IMAGE_SIZE = 224
DEFAULT_SEED = 42
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_MIN_TUMOR_PIXELS = 1
DEFAULT_THRESHOLD = 0.5

MODALITY_NAMES = (
    "t1",
    "t1ce",
    "t2",
    "flair",
)


@dataclass(frozen=True)
class EvaluationProtocol:
    """
    Frozen evaluation protocol for ARMT-GAN scientific evaluation.
    
    This protocol is immutable. Any change requires a new protocol version
    and must be recorded in memory.md with a new decision ID.
    """
    # Data
    image_size: int = DEFAULT_IMAGE_SIZE
    min_tumor_pixels: int = DEFAULT_MIN_TUMOR_PIXELS
    
    # Split fractions (must sum to 1.0)
    train_fraction: float = DEFAULT_TRAIN_FRACTION
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    test_fraction: float = DEFAULT_TEST_FRACTION
    
    # Metric computation
    threshold: float = DEFAULT_THRESHOLD
    
    # Reproducibility
    seed: int = DEFAULT_SEED
    device: str = "auto"
    
    def __post_init__(self) -> None:
        # Validate fractions sum to 1.0
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    
    def to_dict(self) -> dict:
        return {
            "image_size": self.image_size,
            "min_tumor_pixels": self.min_tumor_pixels,
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "threshold": self.threshold,
            "seed": self.seed,
            "device": self.device,
        }


BASELINE_PROTOCOL = EvaluationProtocol()


@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable evaluation configuration for a specific run."""
    protocol: EvaluationProtocol = BASELINE_PROTOCOL
    data_dir: str = ""
    checkpoint: str = ""
    output_dir: str = ""
    save_predictions: bool = True
    
    def to_dict(self) -> dict:
        return {
            "protocol": self.protocol.to_dict(),
            "data_dir": self.data_dir,
            "checkpoint": self.checkpoint,
            "output_dir": self.output_dir,
            "save_predictions": self.save_predictions,
        }


BASELINE_EVAL_CONFIG = EvaluationConfig(protocol=BASELINE_PROTOCOL)


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    directory: str
    modalities: tuple[str, str, str, str]
    segmentation: str


@dataclass
class MetricSummary:
    mean: float
    median: float
    std: float


@dataclass
class EvaluationResult:
    checkpoint: str
    device: str
    seed: int
    image_size: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    total_labeled_patients: int
    train_patients: int
    validation_patients: int
    test_patients: int
    evaluated_patients: int
    evaluated_slices: int
    skipped_slices: int
    dice: MetricSummary
    iou: MetricSummary
    precision: MetricSummary
    sensitivity: MetricSummary
    patient_dice: MetricSummary
    patient_iou: MetricSummary
    patient_precision: MetricSummary
    patient_sensitivity: MetricSummary
    total_inference_seconds: float
    mean_batch_latency_ms: float
    mean_slice_latency_ms: float
    checkpoint_metadata: dict[str, Any]
    raw_artifacts: list[dict]


def set_seed(seed: int) -> None:
    """Make split/evaluation behavior deterministic where practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is preferable for a baseline evaluator.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_patient_modalities(patient_dir: Path) -> tuple[dict[str, Path], Path] | None:
    """
    Find the four required MRI modalities and a segmentation file.

    The implementation intentionally follows the BraTS naming conventions
    used by the training pipeline and does not guess that unrelated files
    are segmentation labels.
    """
    files = list(patient_dir.iterdir())

    modality_paths: dict[str, Path] = {}

    for name, suffix in zip(MODALITY_NAMES, MODALITY_SUFFIXES):
        matches = [
            path
            for path in files
            if path.name.lower().endswith(suffix)
        ]

        if len(matches) != 1:
            # Try .nii.gz convention
            gz_suffix = MODALITY_GZ_SUFFIXES[MODALITY_NAMES.index(name)]
            matches = [
                path
                for path in files
                if path.name.lower().endswith(gz_suffix)
            ]
            if len(matches) != 1:
                # Try new convention
                new_suffix = MODALITY_NEW_SUFFIXES[MODALITY_NAMES.index(name)]
                matches = [
                    path
                    for path in files
                    if path.name.lower().endswith(new_suffix)
                ]
                if len(matches) != 1:
                    return None

        modality_paths[name] = matches[0]

    segmentation_matches = [
        path
        for path in files
        if any(path.name.lower().endswith(suffix)
               for suffix in SEGMENTATION_SUFFIXES)
    ]

    if len(segmentation_matches) != 1:
        return None

    return modality_paths, segmentation_matches[0]


def discover_complete_patients(data_dir: Path) -> list[PatientRecord]:
    """
    Discover complete labeled BraTS patients recursively.

    Only patients containing exactly one recognized copy of every required
    modality and one recognized segmentation file are included.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    patient_dirs = sorted(
        path for path in data_dir.rglob("*")
        if path.is_dir()
    )

    records: list[PatientRecord] = []

    for patient_dir in patient_dirs:
        found = find_patient_modalities(patient_dir)

        if found is None:
            continue

        modality_paths, segmentation = found

        records.append(
            PatientRecord(
                patient_id=patient_dir.name,
                directory=str(patient_dir),
                modalities=(
                    str(modality_paths["t1"]),
                    str(modality_paths["t1ce"]),
                    str(modality_paths["t2"]),
                    str(modality_paths["flair"]),
                ),
                segmentation=str(segmentation),
            )
        )

    # Patient IDs must be unique for a trustworthy patient-level split.
    patient_ids = [record.patient_id for record in records]

    if len(patient_ids) != len(set(patient_ids)):
        raise RuntimeError(
            "Duplicate patient IDs were discovered. "
            "Refusing to create a patient-level split."
        )

    if not records:
        raise RuntimeError(
            f"No complete labeled BraTS patients found under {data_dir}"
        )

    return records


def split_patients_three_way(
    patients: list[PatientRecord],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[PatientRecord], list[PatientRecord], list[PatientRecord]]:
    """
    Reproduce the deterministic patient-level three-way split.

    The list is sorted before shuffling so filesystem traversal order cannot
    change the split. Fractions must sum to 1.0.
    """
    if not (0.0 < train_fraction < 1.0 and 0.0 < validation_fraction < 1.0 and 0.0 < test_fraction < 1.0):
        raise ValueError("All fractions must be between 0 and 1.")
    
    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    patients = sorted(patients, key=lambda record: record.patient_id)

    rng = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_count = max(1, int(round(n * train_fraction)))
    val_count = max(1, int(round(n * validation_fraction)))
    
    # Adjust to ensure all patients are allocated
    if train_count + val_count >= n:
        val_count = n - train_count - 1
        if val_count < 1:
            raise RuntimeError("Not enough patients for three-way split")

    training_patients = shuffled[:train_count]
    validation_patients = shuffled[train_count:train_count + val_count]
    test_patients = shuffled[train_count + val_count:]

    # Verify no patient leakage
    train_ids = {record.patient_id for record in training_patients}
    val_ids = {record.patient_id for record in validation_patients}
    test_ids = {record.patient_id for record in test_patients}

    if train_ids & val_ids:
        raise RuntimeError("Patient leakage detected between train and validation.")
    if train_ids & test_ids:
        raise RuntimeError("Patient leakage detected between train and test.")
    if val_ids & test_ids:
        raise RuntimeError("Patient leakage detected between validation and test.")

    return training_patients, validation_patients, test_patients


def split_patients(
    patients: list[PatientRecord],
    validation_fraction: float,
    seed: int,
) -> tuple[list[PatientRecord], list[PatientRecord]]:
    """
    Two-way split (backward compatible).
    
    Reproduce the deterministic patient-level split used by the baseline.
    The list is sorted before shuffling so filesystem traversal order cannot
    change the split.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    patients = sorted(patients, key=lambda record: record.patient_id)

    rng = random.Random(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    validation_count = max(
        1,
        int(round(len(shuffled) * validation_fraction)),
    )

    validation_patients = shuffled[:validation_count]
    training_patients = shuffled[validation_count:]

    train_ids = {record.patient_id for record in training_patients}
    validation_ids = {record.patient_id for record in validation_patients}

    if train_ids & validation_ids:
        raise RuntimeError("Patient leakage detected between train and validation.")

    return training_patients, validation_patients


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI volume as float32."""
    image = nib.load(path)
    return np.asarray(image.get_fdata(), dtype=np.float32)


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    """
    Z-score normalize non-zero tissue while preserving background as zero.
    """
    volume = volume.astype(np.float32, copy=False)

    nonzero = volume != 0

    if not np.any(nonzero):
        return np.zeros_like(volume, dtype=np.float32)

    values = volume[nonzero]
    mean_value = float(values.mean())
    std_value = float(values.std())

    normalized = np.zeros_like(volume, dtype=np.float32)

    if std_value < 1e-8:
        normalized[nonzero] = values - mean_value
    else:
        normalized[nonzero] = (
            values - mean_value
        ) / std_value

    return normalized


def resize_image_slice(
    image_slice: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Resize one MRI slice with bilinear interpolation."""
    image = Image.fromarray(image_slice.astype(np.float32), mode="F")
    image = image.resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(image, dtype=np.float32)


def resize_mask_slice(
    mask_slice: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Resize a binary segmentation slice with nearest-neighbor."""
    mask = Image.fromarray(mask_slice.astype(np.uint8), mode="L")
    mask = mask.resize(
        (image_size, image_size),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(mask, dtype=np.float32)


def load_patient_slices(
    patient: PatientRecord,
    image_size: int,
    min_tumor_pixels: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """
    Convert one 3D labeled patient into tumor-containing 2D axial samples.

    Returns:
        [(image_tensor, mask_tensor, slice_index), ...]
    """
    # Use shared preprocessing
    preprocessor = BraTSPreprocessor(
        PreprocessingConfig(image_size=image_size)
    )
    
    # Build modality paths dict
    modality_paths = {
        "t1": Path(patient.modalities[0]),
        "t1ce": Path(patient.modalities[1]),
        "t2": Path(patient.modalities[2]),
        "flair": Path(patient.modalities[3]),
    }
    segmentation_path = Path(patient.segmentation)
    
    # Load segmentation to find tumor slices
    segmentation = preprocessor.load_nifti(segmentation_path)
    segmentation = (segmentation > 0).astype(np.float32)
    
    slices: list[tuple[np.ndarray, np.ndarray, int]] = []
    depth = segmentation.shape[2]
    
    for slice_index in range(depth):
        mask_slice = segmentation[:, :, slice_index]
        tumor_pixels = int(mask_slice.sum())
        
        if tumor_pixels < min_tumor_pixels:
            continue
        
        # Preprocess using shared pipeline
        image_tensor = preprocessor.preprocess_modalities(modality_paths, slice_index)
        mask_tensor = preprocessor.preprocess_segmentation(segmentation_path, slice_index)
        
        # Convert to numpy for evaluation metrics
        image_np = image_tensor.numpy().astype(np.float32)  # [4, H, W]
        mask_np = mask_tensor.numpy().astype(np.float32)    # [1, H, W]
        
        slices.append((image_np, mask_np, slice_index))
    
    return slices


def dice_score(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return float((2.0 * intersection) / denominator)


def iou_score(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def precision_score(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = np.logical_and(prediction, target).sum()
    false_positive = np.logical_and(prediction, ~target).sum()

    denominator = true_positive + false_positive

    if denominator == 0:
        return 1.0 if target.sum() == 0 else 0.0

    return float(true_positive / denominator)


def sensitivity_score(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = np.logical_and(prediction, target).sum()
    false_negative = np.logical_and(~prediction, target).sum()

    denominator = true_positive + false_negative

    if denominator == 0:
        return 1.0

    return float(true_positive / denominator)


def summarize(values: list[float]) -> MetricSummary:
    if not values:
        raise ValueError("Cannot summarize an empty metric list.")

    return MetricSummary(
        mean=float(mean(values)),
        median=float(median(values)),
        std=float(stdev(values)) if len(values) > 1 else 0.0,
    )


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Load only the current-format checkpoint.

    We intentionally reject the historical checkpoint format rather than
    silently treating it as the official baseline.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Checkpoint is not a dictionary. "
            "Refusing to evaluate it as a current-format baseline."
        )

    required_keys = {
        "model_state_dict",
        "epoch",
        "seed",
        "image_size",
        "in_channels",
        "out_channels",
        "validation_metrics",
    }

    missing = required_keys - checkpoint.keys()

    if missing:
        raise RuntimeError(
            "Checkpoint is not in the current baseline format. "
            f"Missing keys: {sorted(missing)}"
        )

    if checkpoint["in_channels"] != 4:
        raise RuntimeError(
            f"Expected 4 input channels, got {checkpoint['in_channels']}"
        )

    if checkpoint["out_channels"] != 1:
        raise RuntimeError(
            f"Expected 1 output channel, got {checkpoint['out_channels']}"
        )

    if checkpoint["image_size"] != DEFAULT_IMAGE_SIZE:
        raise RuntimeError(
            "Checkpoint image size does not match evaluator default: "
            f"{checkpoint['image_size']} != {DEFAULT_IMAGE_SIZE}"
        )

    model = ARMTGenerator2D(
        in_channels=4,
        out_channels=1,
        features=[64, 128, 256],
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    metadata = {
        "epoch": checkpoint["epoch"],
        "seed": checkpoint["seed"],
        "image_size": checkpoint["image_size"],
        "in_channels": checkpoint["in_channels"],
        "out_channels": checkpoint["out_channels"],
        "validation_metrics": checkpoint["validation_metrics"],
    }

    return model, metadata


def evaluate(
    model: nn.Module,
    patients: list[PatientRecord],
    device: torch.device,
    protocol: EvaluationProtocol,
    output_dir: Path | None = None,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    dict[str, dict[str, float]],
    int,
    int,
    float,
    float,
    list[dict],  # Raw prediction artifacts for provenance
]:
    """
    Evaluate every tumor-containing slice in the held-out patients.

    Returns slice-level metric lists, patient-level metrics, counts and
    latency measurements, plus raw prediction artifacts.
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

    raw_artifacts: list[dict] = []  # For provenance

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
            image = torch.from_numpy(image_np).unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.inference_mode():
                prediction = model(image)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            total_inference_seconds += elapsed
            inference_calls += 1

            # Raw probability mask [H, W]
            pred_prob = prediction.squeeze(0).squeeze(0).detach().cpu().numpy()
            # Binary prediction at threshold
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

            # Save raw prediction artifact
            if output_dir is not None:
                pred_filename = f"{patient.patient_id}_slice{slice_index:03d}_pred.npy"
                pred_path = pred_dir / pred_filename
                np.save(pred_path, pred_prob.astype(np.float32))
                
                raw_artifacts.append({
                    "patient_id": patient.patient_id,
                    "slice_index": slice_index,
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


def build_result(
    checkpoint_path: Path,
    device: torch.device,
    protocol: EvaluationProtocol,
    all_patients: list[PatientRecord],
    training_patients: list[PatientRecord],
    validation_patients: list[PatientRecord],
    test_patients: list[PatientRecord],
    slice_metrics: tuple[
        list[float],
        list[float],
        list[float],
        list[float],
    ],
    patient_metrics: dict[str, dict[str, float]],
    evaluated_slices: int,
    skipped_slices: int,
    total_inference_seconds: float,
    mean_slice_latency_ms: float,
    checkpoint_metadata: dict[str, Any],
    raw_artifacts: list[dict],
) -> EvaluationResult:
    (
        slice_dice,
        slice_iou,
        slice_precision,
        slice_sensitivity,
    ) = slice_metrics

    patient_dice = [
        metrics["dice"]
        for metrics in patient_metrics.values()
    ]

    patient_iou = [
        metrics["iou"]
        for metrics in patient_metrics.values()
    ]

    patient_precision = [
        metrics["precision"]
        for metrics in patient_metrics.values()
    ]

    patient_sensitivity = [
        metrics["sensitivity"]
        for metrics in patient_metrics.values()
    ]

    return EvaluationResult(
        checkpoint=str(checkpoint_path),
        device=str(device),
        seed=protocol.seed,
        image_size=protocol.image_size,
        train_fraction=protocol.train_fraction,
        validation_fraction=protocol.validation_fraction,
        test_fraction=protocol.test_fraction,
        total_labeled_patients=len(all_patients),
        train_patients=len(training_patients),
        validation_patients=len(validation_patients),
        test_patients=len(test_patients),
        evaluated_patients=len(patient_metrics),
        evaluated_slices=evaluated_slices,
        skipped_slices=skipped_slices,
        dice=summarize(slice_dice),
        iou=summarize(slice_iou),
        precision=summarize(slice_precision),
        sensitivity=summarize(slice_sensitivity),
        patient_dice=summarize(patient_dice),
        patient_iou=summarize(patient_iou),
        patient_precision=summarize(patient_precision),
        patient_sensitivity=summarize(patient_sensitivity),
        total_inference_seconds=total_inference_seconds,
        mean_batch_latency_ms=mean_slice_latency_ms,
        mean_slice_latency_ms=mean_slice_latency_ms,
        checkpoint_metadata=checkpoint_metadata,
        raw_artifacts=raw_artifacts,
    )


def serialize_result(result: EvaluationResult) -> dict[str, Any]:
    return asdict(result)


def print_result(result: EvaluationResult) -> None:
    print("\n" + "=" * 72)
    print("ARMT-GAN BraTS BASELINE EVALUATION")
    print("=" * 72)

    print(f"Checkpoint:             {result.checkpoint}")
    print(f"Device:                 {result.device}")
    print(f"Seed:                   {result.seed}")
    print(f"Image size:             {result.image_size}x{result.image_size}")
    print(f"Train fraction:         {result.train_fraction:.2f}")
    print(f"Validation fraction:    {result.validation_fraction:.2f}")
    print(f"Test fraction:          {result.test_fraction:.2f}")

    print("\nPATIENT SPLIT")
    print(f"Complete labeled:       {result.total_labeled_patients}")
    print(f"Training patients:      {result.train_patients}")
    print(f"Validation patients:    {result.validation_patients}")
    print(f"Test patients:          {result.test_patients}")
    print(f"Evaluated patients:     {result.evaluated_patients}")

    print("\nSAMPLES")
    print(f"Evaluated slices:       {result.evaluated_slices}")
    print(f"Skipped slices:         {result.skipped_slices}")

    print("\nSLICE-LEVEL METRICS")
    print(
        f"Dice        mean={result.dice.mean:.6f} "
        f"median={result.dice.median:.6f} "
        f"std={result.dice.std:.6f}"
    )
    print(
        f"IoU         mean={result.iou.mean:.6f} "
        f"median={result.iou.median:.6f} "
        f"std={result.iou.std:.6f}"
    )
    print(
        f"Precision   mean={result.precision.mean:.6f} "
        f"median={result.precision.median:.6f} "
        f"std={result.precision.std:.6f}"
    )
    print(
        f"Sensitivity mean={result.sensitivity.mean:.6f} "
        f"median={result.sensitivity.median:.6f} "
        f"std={result.sensitivity.std:.6f}"
    )

    print("\nPATIENT-LEVEL METRICS")
    print(
        f"Dice        mean={result.patient_dice.mean:.6f} "
        f"median={result.patient_dice.median:.6f} "
        f"std={result.patient_dice.std:.6f}"
    )
    print(
        f"IoU         mean={result.patient_iou.mean:.6f} "
        f"median={result.patient_iou.median:.6f} "
        f"std={result.patient_iou.std:.6f}"
    )
    print(
        f"Precision   mean={result.patient_precision.mean:.6f} "
        f"median={result.patient_precision.median:.6f} "
        f"std={result.patient_precision.std:.6f}"
    )
    print(
        f"Sensitivity mean={result.patient_sensitivity.mean:.6f} "
        f"median={result.patient_sensitivity.median:.6f} "
        f"std={result.patient_sensitivity.std:.6f}"
    )

    print("\nLATENCY")
    print(
        f"Total inference:        "
        f"{result.total_inference_seconds:.4f} seconds"
    )
    print(
        f"Mean batch latency:     "
        f"{result.mean_batch_latency_ms:.4f} ms"
    )
    print(
        f"Mean slice latency:     "
        f"{result.mean_slice_latency_ms:.4f} ms"
    )

    print(f"\nRAW PREDICTION ARTIFACTS: {len(result.raw_artifacts)} files saved")

    print("\nCHECKPOINT METADATA")
    for key, value in result.checkpoint_metadata.items():
        print(f"{key}: {value}")

    print("\nINTERPRETATION")
    print(
        "This is a held-out test evaluation from the labeled "
        "BraTS training cohort. It is NOT an independent test-set result."
    )
    print(
        "No clinical diagnostic or clinical-validation claim is implied."
    )

    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a current-format ARMT-GAN BraTS checkpoint."
    )

    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root directory containing the complete labeled BraTS patients.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Current-format ARMT-GAN generator checkpoint.",
    )

    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path for a JSON experiment record.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional directory for raw prediction artifacts.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=BASELINE_PROTOCOL.seed,
        help="Patient split seed. Default: 42.",
    )

    parser.add_argument(
        "--train_fraction",
        type=float,
        default=BASELINE_PROTOCOL.train_fraction,
        help="Training patient fraction. Default: 0.70.",
    )

    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=BASELINE_PROTOCOL.validation_fraction,
        help="Validation patient fraction. Default: 0.15.",
    )

    parser.add_argument(
        "--test_fraction",
        type=float,
        default=BASELINE_PROTOCOL.test_fraction,
        help="Test patient fraction. Default: 0.15.",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=BASELINE_PROTOCOL.image_size,
        help="Evaluation image size. Default: 224.",
    )

    parser.add_argument(
        "--min_tumor_pixels",
        type=int,
        default=BASELINE_PROTOCOL.min_tumor_pixels,
        help="Minimum original-mask tumor pixels required for a slice.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=BASELINE_PROTOCOL.threshold,
        help="Binary prediction threshold. Default: 0.5.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=BASELINE_PROTOCOL.device,
        choices=["auto", "cpu", "cuda"],
        help="Evaluation device. Defaults to CUDA when available.",
    )

    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Save raw prediction artifacts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build protocol from args
    protocol = EvaluationProtocol(
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        image_size=args.image_size,
        min_tumor_pixels=args.min_tumor_pixels,
        threshold=args.threshold,
        device=args.device,
    )

    set_seed(protocol.seed)

    # Resolve device
    if protocol.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was explicitly requested but is not available."
        )

    device = torch.device(
        "cuda" if (protocol.device == "cuda" or (protocol.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    output_dir = args.output_dir if args.save_predictions else None

    print("Discovering complete labeled BraTS patients...")

    all_patients = discover_complete_patients(args.data_dir)

    training_patients, validation_patients, test_patients = split_patients_three_way(
        patients=all_patients,
        train_fraction=protocol.train_fraction,
        validation_fraction=protocol.validation_fraction,
        test_fraction=protocol.test_fraction,
        seed=protocol.seed,
    )

    train_ids = {patient.patient_id for patient in training_patients}
    val_ids = {patient.patient_id for patient in validation_patients}
    test_ids = {patient.patient_id for patient in test_patients}

    if train_ids & val_ids:
        raise RuntimeError("Fatal patient overlap detected between train and validation.")
    if train_ids & test_ids:
        raise RuntimeError("Fatal patient overlap detected between train and test.")
    if val_ids & test_ids:
        raise RuntimeError("Fatal patient overlap detected between validation and test.")

    print(f"Complete labeled patients: {len(all_patients)}")
    print(f"Training split:            {len(training_patients)}")
    print(f"Validation split:          {len(validation_patients)}")
    print(f"Test split:                {len(test_patients)}")

    model, checkpoint_metadata = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    if checkpoint_metadata["seed"] != protocol.seed:
        raise RuntimeError(
            "Checkpoint seed does not match evaluator seed: "
            f"{checkpoint_metadata['seed']} != {protocol.seed}"
        )

    if checkpoint_metadata["image_size"] != protocol.image_size:
        raise RuntimeError(
            "Checkpoint image size does not match evaluator config: "
            f"{checkpoint_metadata['image_size']} != {protocol.image_size}"
        )

    (
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
    ) = evaluate(
        model=model,
        patients=test_patients,  # Evaluate on TEST split
        device=device,
        protocol=protocol,
        output_dir=output_dir,
    )

    if not patient_metrics:
        raise RuntimeError(
            "No test patients produced evaluable tumor-containing slices."
        )

    result = build_result(
        checkpoint_path=args.checkpoint,
        device=device,
        protocol=protocol,
        all_patients=all_patients,
        training_patients=training_patients,
        validation_patients=validation_patients,
        test_patients=test_patients,
        slice_metrics=(
            slice_dice,
            slice_iou,
            slice_precision,
            slice_sensitivity,
        ),
        patient_metrics=patient_metrics,
        evaluated_slices=evaluated_slices,
        skipped_slices=skipped_slices,
        total_inference_seconds=total_inference_seconds,
        mean_slice_latency_ms=mean_slice_latency_ms,
        checkpoint_metadata=checkpoint_metadata,
        raw_artifacts=raw_artifacts,
    )

    print_result(result)

    if args.output_json is not None:
        args.output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = serialize_result(result)

        args.output_json.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        print(f"\nJSON experiment record written to: {args.output_json}")


if __name__ == "__main__":
    main()