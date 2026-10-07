"""Add artifact status column and scan processing tracking for transaction consistency.

Revision ID: 005
Revises: 004
Create Date: 2026-10-07
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade():
    # Add status column to artifacts table as nullable first
    # Values: PENDING, COMPLETE, FAILED
    op.add_column('artifacts', sa.Column('status', sa.String(), nullable=True))
    
    # Existing artifact rows represent artifacts already created by the previous implementation
    # They should receive COMPLETE status
    op.execute("UPDATE artifacts SET status = 'COMPLETE' WHERE status IS NULL")
    
    # Now make the column non-null with default for future rows
    op.alter_column('artifacts', 'status', nullable=False, server_default='PENDING')
    
    # Add CHECK constraint to restrict status to valid lifecycle values
    op.create_check_constraint(
        'ck_artifact_status_valid',
        'artifacts',
        "status IN ('PENDING', 'COMPLETE', 'FAILED')"
    )
    
    op.create_index('ix_artifacts_status', 'artifacts', ['status'])
    
    # Add processing_started_at column to scans table to track when processing began
    # This helps distinguish active processing from abandoned/stale attempts
    op.add_column('scans', sa.Column('processing_started_at', sa.DateTime(), nullable=True))
    op.create_index('ix_scans_processing_started_at', 'scans', ['processing_started_at'])


def downgrade():
    op.drop_index('ix_scans_processing_started_at', table_name='scans')
    op.drop_column('scans', 'processing_started_at')
    op.drop_index('ix_artifacts_status', table_name='artifacts')
    op.drop_constraint('ck_artifact_status_valid', 'artifacts', type_='check')
    op.drop_column('artifacts', 'status')