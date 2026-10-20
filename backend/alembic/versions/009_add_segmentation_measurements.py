
"""Add technical 2D segmentation output measurements.

Revision ID: 009
Revises: 008
"""
from alembic import op
import sqlalchemy as sa

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "predictions",
        sa.Column("predicted_mask_pixels", sa.Integer(), nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("slice_coverage_percent", sa.Float(), nullable=True),
    )


def downgrade():
    op.drop_column("predictions", "slice_coverage_percent")
    op.drop_column("predictions", "predicted_mask_pixels")