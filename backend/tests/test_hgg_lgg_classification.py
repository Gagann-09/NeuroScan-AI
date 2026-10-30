"""Focused unit tests for the BraTS HGG/LGG classification pipeline.

Uses lightweight synthetic fixtures. Does not download data, start full
training, or run final test evaluation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from ai_pipeline.datasets.brats_classification import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    BraTSHggLggClassificationDataset,
    ClassificationSplitConfig,
    discover_classification_patients,
    load_brats2020_cohort_labels,
    stratified_patient_split,
    volume_middle_slice_index,
)
from ai_pipeline.models.hgg_lgg_classifier import (
    HggLggClassifier,
    count_trainable_parameters,
)
from ai_pipeline.preprocessing import PreprocessingConfig  # noqa: E402

EXPECTED_CLASS_NAMES = ("HGG", "LGG")
EXPECTED_CLASS_TO_INDEX = {"HGG": 0, "LGG": 1}

# Small synthetic NIfTI volume shape for fast tests.
SYNTHETIC_SHAPE = (16, 16, 8)  # depth 8 -> middle slice index 4


def _write_synthetic_nifti(path: Path, shape: tuple[int, int, int] = SYNTHETIC_SHAPE) -> None:
    """Write a minimal valid NIfTI-1 volume for testing."""
    data = np.random.RandomState(0).randn(*shape).astype(np.float32)
    affine = np.eye(4)
    image = nib.Nifti1Image(data, affine)
    nib.save(image, str(path))


def _create_synthetic_patient(
    tmp_path: Path,
    subject_id: str,
    grade: str,
    modalities: tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
) -> Path:
    """Create a synthetic patient directory with dummy NIfTI modalities."""
    patient_dir = tmp_path / subject_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    for modality in modalities:
        _write_synthetic_nifti(
            patient_dir / f"{subject_id}_{modality}.nii"
        )
    return patient_dir


def _write_mapping_csv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    mapping_path = tmp_path / "name_mapping.csv"
    with mapping_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Grade",
                "BraTS_2017_subject_ID",
                "BraTS_2018_subject_ID",
                "TCGA_TCIA_subject_ID",
                "BraTS_2019_subject_ID",
                "BraTS_2020_subject_ID",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return mapping_path


# ---------------------------------------------------------------------------
# Label mapping tests
# ---------------------------------------------------------------------------

def test_load_brats2020_cohort_labels(tmp_path):
    rows = [
        {"Grade": "HGG", "BraTS_2020_subject_ID": "BraTS20_Training_001"},
        {"Grade": "LGG", "BraTS_2020_subject_ID": "BraTS20_Training_002"},
        {"Grade": "HGG", "BraTS_2020_subject_ID": "BraTS20_Training_003"},
    ]
    mapping_path = _write_mapping_csv(tmp_path, rows)

    labels = load_brats2020_cohort_labels(mapping_path)

    assert labels == {
        "BraTS20_Training_001": "HGG",
        "BraTS20_Training_002": "LGG",
        "BraTS20_Training_003": "HGG",
    }


def test_load_brats2020_cohort_labels_skips_invalid_grades(tmp_path):
    rows = [
        {"Grade": "HGG", "BraTS_2020_subject_ID": "BraTS20_Training_001"},
        {"Grade": "UNKNOWN", "BraTS_2020_subject_ID": "BraTS20_Training_002"},
        {"Grade": "", "BraTS_2020_subject_ID": "BraTS20_Training_003"},
        {"Grade": "LGG", "BraTS_2020_subject_ID": ""},
    ]
    mapping_path = _write_mapping_csv(tmp_path, rows)

    labels = load_brats2020_cohort_labels(mapping_path)

    assert labels == {"BraTS20_Training_001": "HGG"}


def test_class_index_consistency():
    assert CLASS_NAMES == EXPECTED_CLASS_NAMES
    assert CLASS_TO_INDEX == EXPECTED_CLASS_TO_INDEX
    assert CLASS_TO_INDEX["HGG"] == 0
    assert CLASS_TO_INDEX["LGG"] == 1


# ---------------------------------------------------------------------------
# Patient discovery and split tests
# ---------------------------------------------------------------------------

def test_discover_classification_patients_requires_all_modalities(tmp_path):
    # Complete patient.
    _create_synthetic_patient(tmp_path, "BraTS20_Training_001", "HGG")
    # Missing flair.
    patient_dir = tmp_path / "BraTS20_Training_002"
    patient_dir.mkdir(parents=True, exist_ok=True)
    for modality in ("t1", "t1ce", "t2"):
        _write_synthetic_nifti(patient_dir / f"BraTS20_Training_002_{modality}.nii")

    labels = {
        "BraTS20_Training_001": "HGG",
        "BraTS20_Training_002": "LGG",
    }

    patients = discover_classification_patients(tmp_path, labels)

    assert len(patients) == 1
    assert patients[0]["subject_id"] == "BraTS20_Training_001"
    assert patients[0]["grade"] == "HGG"
    assert patients[0]["class_index"] == 0


def test_discover_classification_patients_requires_label(tmp_path):
    _create_synthetic_patient(tmp_path, "BraTS20_Training_001", "HGG")

    # No labels provided.
    patients = discover_classification_patients(tmp_path, {})
    assert patients == []


def test_stratified_patient_split_no_leakage(tmp_path):
    # Create 30 patients: 20 HGG, 10 LGG.
    for i in range(1, 21):
        _create_synthetic_patient(tmp_path, f"BraTS20_Training_{i:03d}", "HGG")
    for i in range(21, 31):
        _create_synthetic_patient(tmp_path, f"BraTS20_Training_{i:03d}", "LGG")

    labels = {
        f"BraTS20_Training_{i:03d}": ("HGG" if i <= 20 else "LGG")
        for i in range(1, 31)
    }
    patients = discover_classification_patients(tmp_path, labels)
    assert len(patients) == 30

    config = ClassificationSplitConfig(seed=42)
    splits = stratified_patient_split(patients, config=config)

    train_ids = {p["subject_id"] for p in splits["train"]}
    val_ids = {p["subject_id"] for p in splits["validation"]}
    test_ids = {p["subject_id"] for p in splits["test"]}

    # No patient in multiple partitions.
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    # All patients assigned exactly once.
    assert train_ids | val_ids | test_ids == set(p["subject_id"] for p in patients)
    # Fractions approximately correct.
    total = len(patients)
    assert len(splits["train"]) == pytest.approx(0.70 * total, abs=2)
    assert len(splits["validation"]) == pytest.approx(0.15 * total, abs=1)
    assert len(splits["test"]) == pytest.approx(0.15 * total, abs=1)


def test_stratified_patient_split_preserves_grade_ratio(tmp_path):
    for i in range(1, 31):
        grade = "HGG" if i <= 20 else "LGG"
        _create_synthetic_patient(tmp_path, f"BraTS20_Training_{i:03d}", grade)

    labels = {
        f"BraTS20_Training_{i:03d}": ("HGG" if i <= 20 else "LGG")
        for i in range(1, 31)
    }
    patients = discover_classification_patients(tmp_path, labels)

    config = ClassificationSplitConfig(seed=42)
    splits = stratified_patient_split(patients, config=config)

    for partition_name in ("train", "validation", "test"):
        partition = splits[partition_name]
        hgg_count = sum(1 for p in partition if p["grade"] == "HGG")
        lgg_count = sum(1 for p in partition if p["grade"] == "LGG")
        # Both classes should be present in every partition (stratification).
        assert hgg_count > 0, f"No HGG in {partition_name}"
        assert lgg_count > 0, f"No LGG in {partition_name}"


# ---------------------------------------------------------------------------
# Slice selection tests
# ---------------------------------------------------------------------------

def test_volume_middle_slice_index_deterministic(tmp_path):
    _create_synthetic_patient(tmp_path, "BraTS20_Training_001", "HGG")
    patient_dir = tmp_path / "BraTS20_Training_001"

    index1 = volume_middle_slice_index(patient_dir)
    index2 = volume_middle_slice_index(patient_dir)

    assert index1 == index2
    assert index1 == SYNTHETIC_SHAPE[2] // 2  # 8 // 2 = 4


def test_dataset_output_shape_and_modality_order(tmp_path):
    _create_synthetic_patient(tmp_path, "BraTS20_Training_001", "HGG")
    labels = {"BraTS20_Training_001": "HGG"}
    patients = discover_classification_patients(tmp_path, labels)
    assert len(patients) == 1

    config = PreprocessingConfig(image_size=224)
    dataset = BraTSHggLggClassificationDataset(patients, config=config)

    image, target, metadata = dataset[0]

    # Output shape [4, 224, 224].
    assert image.shape == (4, 224, 224)
    assert image.dtype == torch.float32

    # Target is correct class index.
    assert target == 0  # HGG

    # Metadata recorded.
    assert metadata["subject_id"] == "BraTS20_Training_001"
    assert metadata["grade"] == "HGG"
    assert metadata["class_index"] == 0
    assert "slice_index" in metadata
    assert metadata["preprocessing_config"]["image_size"] == 224
    assert metadata["preprocessing_config"]["modality_order"] == [
        "t1",
        "t1ce",
        "t2",
        "flair",
    ]


def test_dataset_deterministic_across_calls(tmp_path):
    _create_synthetic_patient(tmp_path, "BraTS20_Training_001", "HGG")
    labels = {"BraTS20_Training_001": "HGG"}
    patients = discover_classification_patients(tmp_path, labels)

    config = PreprocessingConfig(image_size=224)
    dataset = BraTSHggLggClassificationDataset(patients, config=config)

    image1, target1, _ = dataset[0]
    image2, target2, _ = dataset[0]

    assert torch.equal(image1, image2)
    assert target1 == target2


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

def test_model_output_shape():
    model = HggLggClassifier()
    batch = torch.randn(2, 4, 224, 224)
    logits = model(batch)
    assert logits.shape == (2, 2)
    assert logits.dtype == torch.float32


def test_model_rejects_wrong_channels():
    model = HggLggClassifier()
    batch = torch.randn(2, 1, 224, 224)
    with pytest.raises(ValueError, match="four input channels"):
        model(batch)


def test_model_rejects_wrong_dimensions():
    model = HggLggClassifier()
    batch = torch.randn(4, 224, 224)
    with pytest.raises(ValueError, match="4D input"):
        model(batch)


def test_parameter_count():
    model = HggLggClassifier()
    count = count_trainable_parameters(model)
    assert count > 0
    assert isinstance(count, int)


# ---------------------------------------------------------------------------
# Metric calculation tests on known predictions
# ---------------------------------------------------------------------------

def test_balanced_accuracy_known_case():
    from ai_pipeline.training.train_hgg_lgg_classifier import (
        balanced_accuracy,
        lgg_sensitivity,
        hgg_sensitivity,
        confusion_matrix,
    )

    # Perfect predictions: 2 HGG, 2 LGG.
    predictions = torch.tensor([0, 0, 1, 1])
    targets = torch.tensor([0, 0, 1, 1])
    assert balanced_accuracy(predictions, targets) == 1.0
    assert lgg_sensitivity(predictions, targets) == 1.0
    assert hgg_sensitivity(predictions, targets) == 1.0
    assert confusion_matrix(predictions, targets) == [[2, 0], [0, 2]]

    # All wrong: predict HGG for everything.
    predictions = torch.tensor([0, 0, 0, 0])
    targets = torch.tensor([0, 0, 1, 1])
    assert balanced_accuracy(predictions, targets) == 0.5  # (1.0 + 0.0) / 2
    assert lgg_sensitivity(predictions, targets) == 0.0
    assert hgg_sensitivity(predictions, targets) == 1.0
    assert confusion_matrix(predictions, targets) == [[2, 0], [2, 0]]


def test_majority_class_baseline_ordinary_accuracy_held_out_distribution():
    """Always-HGG baseline on the held-out distribution (44 HGG, 12 LGG).

    ordinary_accuracy must compare predictions against true labels, not
    against themselves. For 44 HGG and 12 LGG subjects, always predicting
    HGG yields ordinary accuracy 44/56 (approx 0.785714), not 1.0.
    The other baseline metrics are preserved:
        balanced_accuracy == 0.5
        lgg_sensitivity == 0.0
        hgg_sensitivity == 1.0
    """
    from ai_pipeline.evaluation.evaluate_hgg_lgg_classifier import (
        majority_class_baseline,
    )

    # Held-out class distribution: 44 HGG (class 0), 12 LGG (class 1).
    targets = np.array([0] * 44 + [1] * 12)

    baseline = majority_class_baseline(targets)

    assert baseline["ordinary_accuracy"] == pytest.approx(44 / 56, abs=1e-6)
    assert baseline["balanced_accuracy"] == pytest.approx(0.5, abs=1e-6)
    assert baseline["lgg_sensitivity"] == pytest.approx(0.0, abs=1e-6)
    assert baseline["hgg_sensitivity"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Configuration reproducibility tests
# ---------------------------------------------------------------------------

def test_split_config_is_frozen():
    config = ClassificationSplitConfig()
    assert config.train_fraction == 0.70
    assert config.validation_fraction == 0.15
    assert config.test_fraction == 0.15
    assert config.seed == 42

    # Frozen dataclass: assignment should raise.
    with pytest.raises(Exception):
        config.train_fraction = 0.80


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        ClassificationSplitConfig(
            train_fraction=0.5,
            validation_fraction=0.3,
            test_fraction=0.1,
        )


def test_split_deterministic_same_seed(tmp_path):
    for i in range(1, 21):
        _create_synthetic_patient(tmp_path, f"BraTS20_Training_{i:03d}", "HGG")
    for i in range(21, 31):
        _create_synthetic_patient(tmp_path, f"BraTS20_Training_{i:03d}", "LGG")

    labels = {
        f"BraTS20_Training_{i:03d}": ("HGG" if i <= 20 else "LGG")
        for i in range(1, 31)
    }
    patients = discover_classification_patients(tmp_path, labels)

    config = ClassificationSplitConfig(seed=42)
    splits_a = stratified_patient_split(patients, config=config)
    splits_b = stratified_patient_split(patients, config=config)

    assert [p["subject_id"] for p in splits_a["train"]] == [
        p["subject_id"] for p in splits_b["train"]
    ]
    assert [p["subject_id"] for p in splits_a["test"]] == [
        p["subject_id"] for p in splits_b["test"]
    ]
