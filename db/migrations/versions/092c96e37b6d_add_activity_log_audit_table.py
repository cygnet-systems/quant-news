"""add activity_log audit table

Revision ID: 092c96e37b6d
Revises: 003
Create Date: 2026-07-25 01:57:57.384143

Autogenerate also proposed NOT NULL tightening on created_at/evaluated_at/
position_size across six pre-existing tables. That is unrelated model-vs-DB
drift, would fail on any existing NULL row, and does not belong in this
migration -- removed deliberately. Fix it separately with a backfill.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '092c96e37b6d'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'activity_log',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('run_id', sa.String(length=36), nullable=False),
        sa.Column('run_title', sa.Text(), nullable=True),
        sa.Column('stage', sa.String(length=32), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_activity_run', 'activity_log', ['run_id'], unique=False)
    op.create_index('ix_activity_user_time', 'activity_log',
                    ['user_id', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_activity_user_time', table_name='activity_log')
    op.drop_index('ix_activity_run', table_name='activity_log')
    op.drop_table('activity_log')
