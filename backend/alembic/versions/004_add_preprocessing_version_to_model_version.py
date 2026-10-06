"""add preprocessing_version to model_version

Revision ID: 004
Revises: 003
Create Date: 2026-10-06 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add preprocessing_version column to model_versions table
    op.add_column('model_versions', sa.Column('preprocessing_version', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove preprocessing_version column
    op.drop_column('model_versions', 'preprocessing_version')