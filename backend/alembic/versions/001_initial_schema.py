"""initial schema

Revision ID: 001
Revises: 
Create Date: 2026-10-06 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create scans table
    op.create_table(
        'scans',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('filename', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('mask_path', sa.String(), nullable=True),
        sa.Column('xai_path', sa.String(), nullable=True),
        sa.Column('report_path', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_scans_id'), 'scans', ['id'], unique=False)

    # Create modality_files table
    op.create_table(
        'modality_files',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('scan_id', sa.String(), nullable=True),
        sa.Column('modality', sa.String(), nullable=True),
        sa.Column('object_path', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['scan_id'], ['scans.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_modality_files_id'), 'modality_files', ['id'], unique=False)
    op.create_index(op.f('ix_modality_files_modality'), 'modality_files', ['modality'], unique=False)
    op.create_index(op.f('ix_modality_files_scan_id'), 'modality_files', ['scan_id'], unique=False)

    # Create predictions table
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('scan_id', sa.String(), nullable=True),
        sa.Column('tumor_detected', sa.Boolean(), nullable=True),
        sa.Column('anomaly_area_cm2', sa.Float(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('who_grade', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['scan_id'], ['scans.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_predictions_id'), 'predictions', ['id'], unique=False)
    op.create_index(op.f('ix_predictions_scan_id'), 'predictions', ['scan_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_predictions_scan_id'), table_name='predictions')
    op.drop_index(op.f('ix_predictions_id'), table_name='predictions')
    op.drop_table('predictions')
    op.drop_index(op.f('ix_modality_files_scan_id'), table_name='modality_files')
    op.drop_index(op.f('ix_modality_files_modality'), table_name='modality_files')
    op.drop_index(op.f('ix_modality_files_id'), table_name='modality_files')
    op.drop_table('modality_files')
    op.drop_index(op.f('ix_scans_id'), table_name='scans')
    op.drop_table('scans')