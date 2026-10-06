"""add model_version artifact tables and prediction provenance

Revision ID: 002
Revises: 001
Create Date: 2026-10-06 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create model_versions table
    op.create_table(
        'model_versions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('checkpoint_path', sa.String(), nullable=False),
        sa.Column('config_hash', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_versions_id'), 'model_versions', ['id'], unique=False)

    # Create artifacts table
    op.create_table(
        'artifacts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('prediction_id', sa.Integer(), nullable=True),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('object_path', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['prediction_id'], ['predictions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_artifacts_id'), 'artifacts', ['id'], unique=False)
    op.create_index(op.f('ix_artifacts_prediction_id'), 'artifacts', ['prediction_id'], unique=False)
    op.create_index(op.f('ix_artifacts_type'), 'artifacts', ['type'], unique=False)

    # Add new columns to scans table
    op.add_column('scans', sa.Column('xai_raw_path', sa.String(), nullable=True))

    # Add new columns to predictions table
    op.add_column('predictions', sa.Column('model_version_id', sa.String(), nullable=True))
    op.add_column('predictions', sa.Column('dice', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('iou', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('created_at', sa.DateTime(), nullable=True))

    # Create foreign key from predictions to model_versions
    op.create_foreign_key(
        'fk_predictions_model_version', 'predictions', 'model_versions',
        ['model_version_id'], ['id']
    )

    # Create foreign key from predictions to scans (replace existing loose FK)
    # Note: The original migration had a FK but it was nullable=True. We keep it nullable for backward compatibility.
    # The existing FK constraint on scan_id is already present from initial migration.
    # No need to recreate it.

    # Make scan_id NOT NULL in predictions for new records (existing records remain)
    # This is handled at application level; we don't alter existing data.

    # Add index for model_version_id on predictions
    op.create_index(op.f('ix_predictions_model_version_id'), 'predictions', ['model_version_id'], unique=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index(op.f('ix_predictions_model_version_id'), table_name='predictions')
    
    # Drop foreign key to model_versions
    op.drop_constraint('fk_predictions_model_version', 'predictions', type_='foreignkey')
    
    # Drop columns from predictions
    op.drop_column('predictions', 'created_at')
    op.drop_column('predictions', 'iou')
    op.drop_column('predictions', 'dice')
    op.drop_column('predictions', 'model_version_id')
    
    # Drop column from scans
    op.drop_column('scans', 'xai_raw_path')
    
    # Drop artifacts table
    op.drop_index(op.f('ix_artifacts_type'), table_name='artifacts')
    op.drop_index(op.f('ix_artifacts_prediction_id'), table_name='artifacts')
    op.drop_index(op.f('ix_artifacts_id'), table_name='artifacts')
    op.drop_table('artifacts')
    
    # Drop model_versions table
    op.drop_index(op.f('ix_model_versions_id'), table_name='model_versions')
    op.drop_table('model_versions')