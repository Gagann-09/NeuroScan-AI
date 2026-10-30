
"""Add BraTS HGG/LGG cohort-classifier fields to predictions.

Revision ID: 011
Revises: 010
"""
from alembic import op
import sqlalchemy as sa

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "predictions",
        sa.Column("hgg_lgg_class", sa.String(), nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("hgg_lgg_probability", sa.Float(), nullable=True),
    )


def downgrade():
    op.drop_column("predictions", "hgg_lgg_probability")
    op.drop_column("predictions", "hgg_lgg_class")
