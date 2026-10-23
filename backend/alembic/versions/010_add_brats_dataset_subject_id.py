
"""Persist optional BraTS dataset subject provenance.

Revision ID: 010
Revises: 009
"""
from alembic import op
import sqlalchemy as sa

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "scans",
        sa.Column("dataset_subject_id", sa.String(length=64), nullable=True),
    )


def downgrade():
    op.drop_column("scans", "dataset_subject_id")