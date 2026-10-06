"""
Database Model Tests

Tests verify the SQLAlchemy model definitions, relationships, and constraints
for the persistence and provenance schema (P6 Batch 3A).
"""
import pytest
from sqlalchemy import inspect
from app.db.models import (
    Scan, ModalityFile, ModelVersion, Prediction, Artifact
)
from app.db.base import Base


class TestModelDefinitions:
    """Test that all required model classes exist and have correct structure."""

    def test_scan_model_exists(self):
        """Scan model should exist with required columns."""
        assert hasattr(Scan, '__tablename__')
        assert Scan.__tablename__ == 'scans'
        
        # Check columns exist
        cols = {c.name for c in Scan.__table__.columns}
        expected = {'id', 'filename', 'status', 'mask_path', 'xai_path', 
                    'report_path', 'xai_raw_path', 'created_at'}
        assert expected.issubset(cols)

    def test_modality_file_model_exists(self):
        """ModalityFile model should exist with required columns."""
        assert hasattr(ModalityFile, '__tablename__')
        assert ModalityFile.__tablename__ == 'modality_files'
        
        cols = {c.name for c in ModalityFile.__table__.columns}
        expected = {'id', 'scan_id', 'modality', 'object_path'}
        assert expected.issubset(cols)

    def test_model_version_model_exists(self):
        """ModelVersion model should exist with required columns."""
        assert hasattr(ModelVersion, '__tablename__')
        assert ModelVersion.__tablename__ == 'model_versions'
        
        cols = {c.name for c in ModelVersion.__table__.columns}
        expected = {'id', 'checkpoint_path', 'config_hash', 'created_at'}
        assert expected.issubset(cols)

    def test_prediction_model_exists(self):
        """Prediction model should exist with required columns."""
        assert hasattr(Prediction, '__tablename__')
        assert Prediction.__tablename__ == 'predictions'
        
        cols = {c.name for c in Prediction.__table__.columns}
        expected = {'id', 'scan_id', 'model_version_id', 'tumor_detected', 
                    'anomaly_area_cm2', 'confidence_score', 'dice', 'iou', 
                    'who_grade', 'created_at'}
        assert expected.issubset(cols)

    def test_artifact_model_exists(self):
        """Artifact model should exist with required columns."""
        assert hasattr(Artifact, '__tablename__')
        assert Artifact.__tablename__ == 'artifacts'
        
        cols = {c.name for c in Artifact.__table__.columns}
        expected = {'id', 'prediction_id', 'type', 'object_path', 'created_at'}
        assert expected.issubset(cols)


class TestScanModel:
    """Test Scan model specifics."""

    def test_scan_primary_key(self):
        """Scan.id should be primary key string."""
        pk_cols = [c for c in Scan.__table__.columns if c.primary_key]
        assert len(pk_cols) == 1
        assert pk_cols[0].name == 'id'
        assert str(pk_cols[0].type) == 'VARCHAR'

    def test_scan_xai_raw_path_column(self):
        """Scan should have xai_raw_path column for raw saliency artifact."""
        col = Scan.__table__.columns['xai_raw_path']
        assert col.nullable is True
        assert str(col.type) == 'VARCHAR'

    def test_scan_created_at_default(self):
        """Scan.created_at should have default."""
        col = Scan.__table__.columns['created_at']
        assert col.default is not None


class TestModalityFileModel:
    """Test ModalityFile model specifics."""

    def test_modality_file_foreign_key(self):
        """ModalityFile.scan_id should have FK to scans.id."""
        fks = list(ModalityFile.__table__.foreign_keys)
        assert len(fks) == 1
        fk = fks[0]
        assert fk.parent.name == 'scan_id'
        assert fk.target_fullname == 'scans.id'

    def test_modality_file_modality_index(self):
        """ModalityFile.modality should be indexed."""
        col = ModalityFile.__table__.columns['modality']
        assert col.index is True


class TestModelVersionModel:
    """Test ModelVersion model specifics."""

    def test_model_version_primary_key(self):
        """ModelVersion.id should be primary key string."""
        pk_cols = [c for c in ModelVersion.__table__.columns if c.primary_key]
        assert len(pk_cols) == 1
        assert pk_cols[0].name == 'id'

    def test_model_version_required_fields(self):
        """ModelVersion.checkpoint_path should be required."""
        col = ModelVersion.__table__.columns['checkpoint_path']
        assert col.nullable is False

    def test_model_version_config_hash_nullable(self):
        """ModelVersion.config_hash should be nullable."""
        col = ModelVersion.__table__.columns['config_hash']
        assert col.nullable is True


class TestPredictionModel:
    """Test Prediction model specifics."""

    def test_prediction_primary_key(self):
        """Prediction.id should be primary key integer."""
        pk_cols = [c for c in Prediction.__table__.columns if c.primary_key]
        assert len(pk_cols) == 1
        assert pk_cols[0].name == 'id'
        assert str(pk_cols[0].type) == 'INTEGER'

    def test_prediction_scan_id_fk(self):
        """Prediction.scan_id should have FK to scans.id."""
        fks = [fk for fk in Prediction.__table__.foreign_keys 
               if fk.parent.name == 'scan_id']
        assert len(fks) == 1
        fk = fks[0]
        assert fk.target_fullname == 'scans.id'

    def test_prediction_model_version_id_fk(self):
        """Prediction.model_version_id should have FK to model_versions.id."""
        fks = [fk for fk in Prediction.__table__.foreign_keys 
               if fk.parent.name == 'model_version_id']
        assert len(fks) == 1
        fk = fks[0]
        assert fk.target_fullname == 'model_versions.id'

    def test_prediction_scan_id_not_null(self):
        """Prediction.scan_id should be NOT NULL."""
        col = Prediction.__table__.columns['scan_id']
        assert col.nullable is False

    def test_prediction_model_version_id_nullable(self):
        """Prediction.model_version_id should be nullable for backward compat."""
        col = Prediction.__table__.columns['model_version_id']
        assert col.nullable is True

    def test_prediction_dice_iou_columns(self):
        """Prediction should have dice and iou columns."""
        dice_col = Prediction.__table__.columns['dice']
        iou_col = Prediction.__table__.columns['iou']
        assert dice_col.nullable is True
        assert iou_col.nullable is True
        assert str(dice_col.type) == 'FLOAT'
        assert str(iou_col.type) == 'FLOAT'

    def test_prediction_created_at(self):
        """Prediction should have created_at."""
        col = Prediction.__table__.columns['created_at']
        assert col.default is not None


class TestArtifactModel:
    """Test Artifact model specifics."""

    def test_artifact_primary_key(self):
        """Artifact.id should be primary key integer."""
        pk_cols = [c for c in Artifact.__table__.columns if c.primary_key]
        assert len(pk_cols) == 1
        assert pk_cols[0].name == 'id'
        assert str(pk_cols[0].type) == 'INTEGER'

    def test_artifact_prediction_id_fk(self):
        """Artifact.prediction_id should have FK to predictions.id."""
        fks = [fk for fk in Artifact.__table__.foreign_keys 
               if fk.parent.name == 'prediction_id']
        assert len(fks) == 1
        fk = fks[0]
        assert fk.target_fullname == 'predictions.id'

    def test_artifact_required_fields(self):
        """Artifact.type and object_path should be required."""
        type_col = Artifact.__table__.columns['type']
        path_col = Artifact.__table__.columns['object_path']
        assert type_col.nullable is False
        assert path_col.nullable is False

    def test_artifact_type_indexed(self):
        """Artifact.type should be indexed."""
        col = Artifact.__table__.columns['type']
        assert col.index is True


class TestMetadataConsistency:
    """Test SQLAlchemy metadata consistency."""

    def test_all_tables_in_metadata(self):
        """All expected tables should be in Base.metadata."""
        tables = set(Base.metadata.tables.keys())
        expected = {'scans', 'modality_files', 'model_versions', 
                    'predictions', 'artifacts'}
        assert expected.issubset(tables)

    def test_table_relationships(self):
        """Verify relationship graph matches design."""
        # scans -> modality_files (one-to-many)
        # scans -> predictions (one-to-one/one-to-many)
        # predictions -> model_versions (many-to-one)
        # predictions -> artifacts (one-to-many)
        
        # Check FK from modality_files to scans
        mf_fks = list(ModalityFile.__table__.foreign_keys)
        assert any(fk.target_fullname == 'scans.id' for fk in mf_fks)
        
        # Check FK from predictions to scans
        pred_fks = list(Prediction.__table__.foreign_keys)
        scan_fks = [fk for fk in pred_fks if fk.target_fullname == 'scans.id']
        mv_fks = [fk for fk in pred_fks if fk.target_fullname == 'model_versions.id']
        assert len(scan_fks) == 1
        assert len(mv_fks) == 1
        
        # Check FK from artifacts to predictions
        art_fks = list(Artifact.__table__.foreign_keys)
        assert any(fk.target_fullname == 'predictions.id' for fk in art_fks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])