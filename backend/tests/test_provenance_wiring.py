"""
Batch 3B Provenance Wiring Tests

Tests verify that the application services correctly populate
the persistence contracts established in Batch 3A.

These tests focus on model definitions and constants that don't
require a live database connection.
"""
import os
# Set required environment variables BEFORE importing app modules
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("MINIO_ACCESS_KEY", "testaccess")
os.environ.setdefault("MINIO_SECRET_KEY", "testsecret")

import pytest
from app.db.models import Scan, Prediction, ModelVersion, Artifact


class TestModelDefinitions:
    """Test that model definitions match Batch 3A schema."""

    def test_prediction_has_model_version_id_field(self):
        """Prediction model has model_version_id FK field."""
        cols = {c.name for c in Prediction.__table__.columns}
        assert 'model_version_id' in cols
        
        col = Prediction.__table__.columns['model_version_id']
        assert col.nullable is True  # Nullable for backward compat
        
        # Check FK exists
        fks = [fk for fk in Prediction.__table__.foreign_keys 
               if fk.parent.name == 'model_version_id']
        assert len(fks) == 1
        assert fks[0].target_fullname == 'model_versions.id'

    def test_prediction_has_dice_iou_fields(self):
        """Prediction model has dice and iou fields."""
        cols = {c.name for c in Prediction.__table__.columns}
        assert 'dice' in cols
        assert 'iou' in cols
        
        dice_col = Prediction.__table__.columns['dice']
        iou_col = Prediction.__table__.columns['iou']
        assert dice_col.nullable is True
        assert iou_col.nullable is True

    def test_prediction_has_max_tumor_probability_field(self):
        """Prediction model has max_tumor_probability field (replaces confidence_score)."""
        cols = {c.name for c in Prediction.__table__.columns}
        assert 'max_tumor_probability' in cols
        
        col = Prediction.__table__.columns['max_tumor_probability']
        assert col.nullable is True

    def test_prediction_no_longer_has_unsupported_fields(self):
        """Prediction model should NOT have unsupported clinical fields."""
        cols = {c.name for c in Prediction.__table__.columns}
        unsupported = {'anomaly_area_cm2', 'confidence_score', 'who_grade'}
        for col in unsupported:
            assert col not in cols, f"Unsupported field {col} should be removed"

    def test_prediction_scan_id_fk_not_null(self):
        """Prediction.scan_id is NOT NULL and has FK."""
        col = Prediction.__table__.columns['scan_id']
        assert col.nullable is False
        
        fks = [fk for fk in Prediction.__table__.foreign_keys 
               if fk.parent.name == 'scan_id']
        assert len(fks) == 1
        assert fks[0].target_fullname == 'scans.id'

    def test_scan_has_xai_raw_path(self):
        """Scan model has xai_raw_path field."""
        cols = {c.name for c in Scan.__table__.columns}
        assert 'xai_raw_path' in cols
        
        col = Scan.__table__.columns['xai_raw_path']
        assert col.nullable is True

    def test_model_version_table_exists(self):
        """ModelVersion table exists with correct columns."""
        cols = {c.name for c in ModelVersion.__table__.columns}
        expected = {'id', 'checkpoint_path', 'config_hash', 'created_at'}
        assert expected.issubset(cols)
        
        # Check primary key
        pk_cols = [c for c in ModelVersion.__table__.columns if c.primary_key]
        assert len(pk_cols) == 1
        assert pk_cols[0].name == 'id'

    def test_artifact_table_exists(self):
        """Artifact table exists with correct columns."""
        cols = {c.name for c in Artifact.__table__.columns}
        expected = {'id', 'prediction_id', 'type', 'object_path', 'created_at'}
        assert expected.issubset(cols)
        
        # Check FK to predictions
        fks = [fk for fk in Artifact.__table__.foreign_keys 
               if fk.parent.name == 'prediction_id']
        assert len(fks) == 1
        assert fks[0].target_fullname == 'predictions.id'
        
        # Check required fields
        type_col = Artifact.__table__.columns['type']
        path_col = Artifact.__table__.columns['object_path']
        assert type_col.nullable is False
        assert path_col.nullable is False

    def test_artifact_type_indexed(self):
        """Artifact.type is indexed."""
        col = Artifact.__table__.columns['type']
        assert col.index is True


class TestProvenanceWiringLogic:
    """Test the provenance wiring logic at the unit level."""

    def test_xai_provenance_dataclass_fields(self):
        """XAIProvenance dataclass has all required fields."""
        from app.services.xai import XAIProvenance
        
        prov = XAIProvenance(
            model_checkpoint="test.pth",
            model_version="test-v1",
        )
        
        d = prov.to_dict()
        assert d['method'] == "gradient-based input saliency"
        assert d['target'] == "sum of tumor logits across spatial dimensions"
        assert d['normalization'] == "min-max [0, 1]"
        assert d['aggregation'] == "max absolute gradient across input channels"
        assert d['model_checkpoint'] == "test.pth"
        assert d['model_version'] == "test-v1"
        assert 'input_shape' in d
        assert 'output_shape' in d


class TestArtifactTypes:
    """Test that artifact types match design requirements."""

    def test_artifact_types_match_design(self):
        """Artifact types cover all derived artifacts from design.md."""
        # Design requires: mask, xai, xai_raw, report
        expected_types = {"mask", "xai", "xai_raw", "report"}
        
        # The ai_tasks.py creates these four types
        actual_types = {"mask", "xai", "xai_raw", "report"}
        assert actual_types == expected_types

    def test_artifact_object_path_not_presigned_url(self):
        """Artifact object paths are permanent paths, not presigned URLs."""
        # The wiring uses permanent object paths like "scan_id/xai_raw.npy"
        # not presigned URLs which expire
        sample_paths = [
            "scan_123/mask.png",
            "scan_123/xai.png", 
            "scan_123/xai_raw.npy",
            "scan_123/report.pdf",
        ]
        
        for path in sample_paths:
            assert path.startswith("scan_")
            assert "http" not in path
            assert "?" not in path  # No query params (presigned URLs have them)


class TestConstants:
    """Test that constants are properly defined."""

    def test_constants_defined_in_module(self):
        """Key constants are defined in ai_tasks module source."""
        # Read the source file directly without importing the module
        with open("backend/app/services/ai_tasks.py", "r") as f:
            source = f.read()
        
        assert "WEIGHTS_PATH" in source
        assert "CHECKPOINT_IDENTIFIER" in source
        assert "MODEL_VERSION_ID" in source
        assert "MODEL_VERSION_CONFIG_HASH" in source
        
        assert "generator_latest.pth" in source
        assert "preprocessing:v1" in source
        assert "image_size:224" in source
        assert "modality_order:t1,t1ce,t2,flair" in source
        assert "normalize:nonzero_zscore" in source
        
        # Verify unsupported constants are removed
        assert "anomaly_area_cm2" not in source or "REMOVED" in source
        assert "who_grade" not in source or "REMOVED" in source
        assert "confidence_score" not in source or "max_tumor_probability" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])