"""Add Kaggle classification tables.

Revision ID: 008
Revises: 007
Create Date: 2026-10-18
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade():
    # kaggle_scans table
    op.create_table(
        'kaggle_scans',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('filename', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True, default='PENDING'),
        sa.Column('image_path', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('processing_started_at', sa.DateTime(), nullable=True),
        sa.Column('user_id', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.firebase_uid'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_kaggle_scans_user_id', 'kaggle_scans', ['user_id'])

    # kaggle_model_versions table
    op.create_table(
        'kaggle_model_versions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('checkpoint_path', sa.String(), nullable=False),
        sa.Column('config_hash', sa.String(), nullable=True),
        sa.Column('preprocessing_version', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # kaggle_predictions table
    op.create_table(
        'kaggle_predictions',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('scan_id', sa.String(), nullable=False),
        sa.Column('model_version_id', sa.String(), nullable=True),
        sa.Column('class_index', sa.Integer(), nullable=True),
        sa.Column('class_name', sa.String(), nullable=True),
        sa.Column('probabilities', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['model_version_id'], ['kaggle_model_versions.id'], ),
        sa.ForeignKeyConstraint(['scan_id'], ['kaggle_scans.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_kaggle_predictions_scan_id', 'kaggle_predictions', ['scan_id'])
    op.create_index('ix_kaggle_predictions_model_version_id', 'kaggle_predictions', ['model_version_id'])


def downgrade():
    op.drop_index('ix_kaggle_predictions_model_version_id', table_name='kaggle_predictions')
    op.drop_index('ix_kaggle_predictions_scan_id', table_name='kaggle_predictions')
    op.drop_table('kaggle_predictions')
    op.drop_table('kaggle_model_versions')
    op.drop_index('ix_kaggle_scans_user_id', table_name='kaggle_scans')
    op.drop_table('kaggle_scans')