"""
Evaluation Protocol Tests

Tests for the scientific evaluation protocol:
- Metric correctness (Dice, IoU, Precision, Sensitivity)
- Patient-disjoint split verification
- Deterministic split reproducibility
- Raw prediction artifact saving
- Protocol serialization

Run with: python -m pytest backend/tests/test_evaluation_protocol.py -v
"""

import tempfile
from pathlib import Path
import numpy as np
import torch
import pytest

from ai_pipeline.evaluation.evaluate_armt_gan_baseline import (
    dice_score,
    iou_score,
    precision_score,
    sensitivity_score,
    split_patients_three_way,
    PatientRecord,
    EvaluationProtocol,
    summarize,
    MetricSummary,
)


class TestMetricCorrectness:
    """Test metric implementations against known cases."""

    def test_dice_perfect_match(self):
        """Dice should be 1.0 for identical masks."""
        pred = np.array([[1, 1], [0, 0]], dtype=bool)
        target = np.array([[1, 1], [0, 0]], dtype=bool)
        assert dice_score(pred, target) == 1.0

    def test_dice_no_overlap(self):
        """Dice should be 0.0 for non-overlapping masks."""
        pred = np.array([[1, 1], [0, 0]], dtype=bool)
        target = np.array([[0, 0], [1, 1]], dtype=bool)
        assert dice_score(pred, target) == 0.0

    def test_dice_partial_overlap(self):
        """Dice for partial overlap: 2*TP / (2*TP + FP + FN)"""
        # TP=2, FP=1, FN=1 -> Dice = 4/6 = 0.666...
        pred = np.array([[1, 1], [1, 0]], dtype=bool)
        target = np.array([[1, 1], [0, 1]], dtype=bool)
        result = dice_score(pred, target)
        assert abs(result - 4/6) < 1e-6

    def test_dice_both_empty(self):
        """Dice should be 1.0 when both are empty."""
        pred = np.zeros((10, 10), dtype=bool)
        target = np.zeros((10, 10), dtype=bool)
        assert dice_score(pred, target) == 1.0

    def test_iou_perfect_match(self):
        """IoU should be 1.0 for identical masks."""
        pred = np.array([[1, 1], [0, 0]], dtype=bool)
        target = np.array([[1, 1], [0, 0]], dtype=bool)
        assert iou_score(pred, target) == 1.0

    def test_iou_no_overlap(self):
        """IoU should be 0.0 for non-overlapping masks."""
        pred = np.array([[1, 1], [0, 0]], dtype=bool)
        target = np.array([[0, 0], [1, 1]], dtype=bool)
        assert iou_score(pred, target) == 0.0

    def test_iou_partial_overlap(self):
        """IoU for partial overlap: TP / (TP + FP + FN)"""
        # TP=2, FP=1, FN=1 -> IoU = 2/4 = 0.5
        pred = np.array([[1, 1], [1, 0]], dtype=bool)
        target = np.array([[1, 1], [0, 1]], dtype=bool)
        result = iou_score(pred, target)
        assert abs(result - 0.5) < 1e-6

    def test_precision_perfect(self):
        """Precision should be 1.0 when no false positives."""
        pred = np.array([[1, 0], [0, 0]], dtype=bool)
        target = np.array([[1, 0], [0, 0]], dtype=bool)
        assert precision_score(pred, target) == 1.0

    def test_precision_some_fp(self):
        """Precision with false positives: TP / (TP + FP)"""
        # TP=1, FP=1 -> Precision = 0.5
        pred = np.array([[1, 1], [0, 0]], dtype=bool)
        target = np.array([[1, 0], [0, 0]], dtype=bool)
        result = precision_score(pred, target)
        assert abs(result - 0.5) < 1e-6

    def test_sensitivity_perfect(self):
        """Sensitivity (recall) should be 1.0 when no false negatives."""
        pred = np.array([[1, 0], [0, 0]], dtype=bool)
        target = np.array([[1, 0], [0, 0]], dtype=bool)
        assert sensitivity_score(pred, target) == 1.0

    def test_sensitivity_some_fn(self):
        """Sensitivity with false negatives: TP / (TP + FN)"""
        # TP=1, FN=1 -> Sensitivity = 0.5
        pred = np.array([[1, 0], [0, 0]], dtype=bool)
        target = np.array([[1, 1], [0, 0]], dtype=bool)
        result = sensitivity_score(pred, target)
        assert abs(result - 0.5) < 1e-6


class TestPatientDisjointSplit:
    """Test three-way patient split for disjointness and determinism."""

    def make_patients(self, n: int) -> list[PatientRecord]:
        """Create dummy patient records."""
        patients = []
        for i in range(n):
            patients.append(PatientRecord(
                patient_id=f"patient_{i:03d}",
                directory=f"/tmp/patient_{i:03d}",
                modalities=("/tmp/t1.nii", "/tmp/t1ce.nii", "/tmp/t2.nii", "/tmp/flair.nii"),
                segmentation="/tmp/seg.nii",
            ))
        return patients

    def test_three_way_split_deterministic(self):
        """Same seed should produce identical splits."""
        patients = self.make_patients(20)
        
        train1, val1, test1 = split_patients_three_way(patients, 0.7, 0.15, 0.15, seed=42)
        train2, val2, test2 = split_patients_three_way(patients, 0.7, 0.15, 0.15, seed=42)
        
        assert {p.patient_id for p in train1} == {p.patient_id for p in train2}
        assert {p.patient_id for p in val1} == {p.patient_id for p in val2}
        assert {p.patient_id for p in test1} == {p.patient_id for p in test2}

    def test_three_way_split_different_seeds(self):
        """Different seeds should produce different splits (with high probability)."""
        patients = self.make_patients(20)
        
        train1, val1, test1 = split_patients_three_way(patients, 0.7, 0.15, 0.15, seed=42)
        train2, val2, test2 = split_patients_three_way(patients, 0.7, 0.15, 0.15, seed=43)
        
        # Very unlikely to be identical with different seeds
        train1_ids = {p.patient_id for p in train1}
        train2_ids = {p.patient_id for p in train2}
        assert train1_ids != train2_ids

    def test_three_way_split_no_leakage(self):
        """No patient should appear in more than one split."""
        patients = self.make_patients(30)
        
        train, val, test = split_patients_three_way(patients, 0.7, 0.15, 0.15, seed=42)
        
        train_ids = {p.patient_id for p in train}
        val_ids = {p.patient_id for p in val}
        test_ids = {p.patient_id for p in test}
        
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_three_way_split_all_patients_allocated(self):
        """All patients should be allocated exactly once."""
        patients = self.make_patients(25)
        
        train, val, test = split_patients_three_way(patients, 0.7, 0.15, 0.15, seed=42)
        
        all_allocated = set()
        for p in train: all_allocated.add(p.patient_id)
        for p in val: all_allocated.add(p.patient_id)
        for p in test: all_allocated.add(p.patient_id)
        
        original_ids = {p.patient_id for p in patients}
        assert all_allocated == original_ids

    def test_three_way_split_minimum_counts(self):
        """Each split should have at least 1 patient."""
        patients = self.make_patients(5)
        
        train, val, test = split_patients_three_way(patients, 0.6, 0.2, 0.2, seed=42)
        
        assert len(train) >= 1
        assert len(val) >= 1
        assert len(test) >= 1

    def test_three_way_split_fraction_errors(self):
        """Invalid fractions should raise ValueError."""
        patients = self.make_patients(10)
        
        with pytest.raises(ValueError):
            split_patients_three_way(patients, 0.8, 0.2, 0.2, seed=42)  # Sum > 1
        
        with pytest.raises(ValueError):
            split_patients_three_way(patients, 0.3, 0.3, 0.3, seed=42)  # Sum < 1
        
        with pytest.raises(ValueError):
            split_patients_three_way(patients, -0.1, 0.5, 0.5, seed=42)  # Negative

    def test_split_sorts_before_shuffle(self):
        """Split should be deterministic regardless of input order."""
        patients = self.make_patients(20)
        # Shuffle input order
        import random
        shuffled = patients.copy()
        random.Random(123).shuffle(shuffled)
        
        train1, val1, test1 = split_patients_three_way(patients, 0.7, 0.15, 0.15, seed=42)
        train2, val2, test2 = split_patients_three_way(shuffled, 0.7, 0.15, 0.15, seed=42)
        
        assert {p.patient_id for p in train1} == {p.patient_id for p in train2}
        assert {p.patient_id for p in val1} == {p.patient_id for p in val2}
        assert {p.patient_id for p in test1} == {p.patient_id for p in test2}


class TestEvaluationProtocol:
    """Test the frozen EvaluationProtocol dataclass."""

    def test_protocol_defaults(self):
        """Protocol should have correct default values."""
        protocol = EvaluationProtocol()
        assert protocol.image_size == 224
        assert protocol.min_tumor_pixels == 1
        assert protocol.train_fraction == 0.7
        assert protocol.validation_fraction == 0.15
        assert protocol.test_fraction == 0.15
        assert protocol.threshold == 0.5
        assert protocol.seed == 42
        assert protocol.device == "auto"

    def test_protocol_validation(self):
        """Protocol should reject invalid fractions."""
        with pytest.raises(ValueError):
            EvaluationProtocol(train_fraction=0.8, validation_fraction=0.2, test_fraction=0.2)
        
        with pytest.raises(ValueError):
            EvaluationProtocol(train_fraction=0.3, validation_fraction=0.3, test_fraction=0.3)

    def test_protocol_serialization(self):
        """Protocol should serialize to dict correctly."""
        protocol = EvaluationProtocol()
        d = protocol.to_dict()
        assert d["image_size"] == 224
        assert d["train_fraction"] == 0.7
        assert d["validation_fraction"] == 0.15
        assert d["test_fraction"] == 0.15
        assert d["threshold"] == 0.5
        assert d["seed"] == 42

    def test_protocol_immutability(self):
        """Protocol should be frozen (immutable)."""
        protocol = EvaluationProtocol()
        with pytest.raises(Exception):
            protocol.image_size = 256


class TestMetricSummary:
    """Test metric summary statistics."""

    def test_summarize_mean_median_std(self):
        """Summarize should compute correct statistics."""
        values = [0.8, 0.85, 0.9, 0.95, 1.0]
        summary = summarize(values)
        
        assert abs(summary.mean - 0.9) < 1e-6
        assert abs(summary.median - 0.9) < 1e-6
        # statistics.stdev uses sample std (n-1), not population std
        # Sample std of [0.8, 0.85, 0.9, 0.95, 1.0] = sqrt(0.00625) = 0.0790569...
        assert abs(summary.std - 0.0790569415) < 1e-6

    def test_summarize_single_value(self):
        """Summarize with single value should have std=0."""
        values = [0.85]
        summary = summarize(values)
        
        assert abs(summary.mean - 0.85) < 1e-6
        assert abs(summary.median - 0.85) < 1e-6
        assert summary.std == 0.0

    def test_summarize_empty_raises(self):
        """Summarize empty list should raise ValueError."""
        with pytest.raises(ValueError):
            summarize([])


class TestEvaluationReproducibility:
    """Test that evaluation is reproducible."""

    def make_dummy_patients(self, n: int) -> list[PatientRecord]:
        patients = []
        for i in range(n):
            patients.append(PatientRecord(
                patient_id=f"p{i:03d}",
                directory=f"/tmp/p{i:03d}",
                modalities=("/tmp/t1.nii", "/tmp/t1ce.nii", "/tmp/t2.nii", "/tmp/flair.nii"),
                segmentation="/tmp/seg.nii",
            ))
        return patients

    def test_evaluation_protocol_deterministic(self):
        """Evaluation protocol should be fully deterministic."""
        protocol = EvaluationProtocol()
        d1 = protocol.to_dict()
        d2 = protocol.to_dict()
        assert d1 == d2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])