"""add user table and scan ownership

Revision ID: 006
Revises: 005
Create Date: 2026-10-10
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('firebase_uid', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_seen_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('firebase_uid')
    )
    op.create_index(op.f('ix_users_firebase_uid'), 'users', ['firebase_uid'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)

    # Add user_id column to scans table (nullable for backward compatibility with existing scans)
    op.add_column('scans', sa.Column('user_id', sa.String(), nullable=True))
    op.create_index(op.f('ix_scans_user_id'), 'scans', ['user_id'], unique=False)
    op.create_foreign_key(
        'fk_scans_user_id', 'scans', 'users',
        ['user_id'], ['firebase_uid']
    )


def downgrade() -> None:
    # Drop foreign key and column from scans
    op.drop_constraint('fk_scans_user_id', 'scans', type_='foreignkey')
    op.drop_index(op.f('ix_scans_user_id'), table_name='scans')
    op.drop_column('scans', 'user_id')

    # Drop users table
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_firebase_uid'), table_name='users')
    op.drop_table('users')