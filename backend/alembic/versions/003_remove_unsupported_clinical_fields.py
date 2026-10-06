"""remove unsupported clinical fields

Revision ID: 003
Revises: 002
Create Date: 2026-10-06 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Remove unsupported clinical fields from predictions table
    op.drop_column('predictions', 'who_grade')
    op.drop_column('predictions', 'anomaly_area_cm2')
    op.drop_column('predictions', 'confidence_score')
    
    # Add max_tumor_probability column
    op.add_column('predictions', sa.Column('max_tumor_probability', sa.Float(), nullable=True))


def downgrade() -> None:
    # Add back the removed columns
    op.add_column('predictions', sa.Column('confidence_score', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('anomaly_area_cm2', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('who_grade', sa.String(), nullable=True))
    
    # Remove the new column
    op.drop_column('predictions', 'max_tumor_probability')