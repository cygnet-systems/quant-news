"""Widen news_articles.impact from VARCHAR(16) to VARCHAR(64).

Revision ID: 003
Revises: 002
"""

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "news_articles",
        "impact",
        type_=sa.String(64),
        existing_type=sa.String(16),
    )


def downgrade() -> None:
    op.alter_column(
        "news_articles",
        "impact",
        type_=sa.String(16),
        existing_type=sa.String(64),
    )
