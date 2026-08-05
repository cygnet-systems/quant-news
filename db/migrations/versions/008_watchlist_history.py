"""Add watchlist_history.

Recent symbol groups lived only in the browser's localStorage, capped at five
entries. That meant a cleared cache, a different browser or a different machine
silently lost every past search, and there was no way to look further back than
the last five. Moving it server-side makes past work recoverable and lets the
Activity page browse it.

The unique index is on COALESCE(owner_uid, '') because Postgres treats NULLs as
distinct: with a plain unique constraint, anonymous sessions would insert a new
row for the same group on every visit instead of bumping use_count.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "watchlist_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("owner_uid", sa.String(length=64), nullable=True),
        sa.Column("symbols_csv", sa.Text(), nullable=False),
        sa.Column("first_used_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("use_count", sa.Integer(), nullable=False, server_default="1"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_watchlist_history_recent", "watchlist_history",
                    ["owner_uid", "last_used_at"])
    op.execute(
        "CREATE UNIQUE INDEX uq_watchlist_history_owner_symbols "
        "ON watchlist_history (COALESCE(owner_uid, ''), symbols_csv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_watchlist_history_owner_symbols")
    op.drop_index("ix_watchlist_history_recent", table_name="watchlist_history")
    op.drop_table("watchlist_history")
