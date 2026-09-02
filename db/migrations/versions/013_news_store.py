"""Durable news store: article timestamps + per-day fetch coverage

Revision ID: 013
Revises: 012
Create Date: 2026-09-02

Point-in-time news used to be cached per (symbol, as_of, window) — a daily
14-day run refetched the same articles every morning. historical_news now
holds every article the vendor returned (with its full timestamp, so the
overnight window can be served from it too) and news_coverage records which
(symbol, day) pairs have been fetched, so a run fetches only the days it has
not seen. Articles older than NEWS_RETENTION_DAYS are pruned.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("historical_news",
                  sa.Column("published_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_historical_news_published_at", "historical_news",
                    ["symbol", "published_at"], unique=False)
    op.create_table(
        "news_coverage",
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("symbol", "day"),
    )


def downgrade() -> None:
    op.drop_table("news_coverage")
    op.drop_index("ix_historical_news_published_at", table_name="historical_news")
    op.drop_column("historical_news", "published_at")
