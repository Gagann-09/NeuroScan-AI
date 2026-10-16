"""Add retry count to scans for stale-pending recovery accounting.

Revision ID: 007
Revises: 006
Create Date: 2026-10-16
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade():
    # attempt_count tracks how many times the stale-PENDING recovery loop has
    # re-dispatched a scan for processing. It is incremented atomically and
    # durably (committed) before each re-dispatch, and capped at
    # MAX_RECOVERY_ATTEMPTS (3). Once the cap is reached the scan is marked
    # FAILED and is not retried again by the recovery loop.
    op.add_column('scans', sa.Column('attempt_count', sa.Integer(), nullable=False, server_default='0'))
    op.create_index('ix_scans_attempt_count', 'scans', ['attempt_count'])


def downgrade():
    op.drop_index('ix_scans_attempt_count', table_name='scans')
    op.drop_column('scans', 'attempt_count')
