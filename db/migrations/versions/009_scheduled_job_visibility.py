"""Add is_public to scheduled_jobs: schedules become per-user shareable.

Every existing job stays public (server_default true), so behavior is
unchanged until someone creates a private schedule. Read-time filtering
lives in scheduler_service.list_jobs: a private job is visible only to its
owner (and Administrators).

Revision ID: 009
Revises: 008
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "scheduled_jobs",
        sa.Column("is_public", sa.Boolean(), nullable=False,
                  server_default=sa.text("true")),
    )


def downgrade() -> None:
    op.drop_column("scheduled_jobs", "is_public")
