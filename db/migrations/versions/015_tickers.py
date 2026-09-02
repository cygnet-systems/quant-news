"""tickers: local symbol lookup cache behind the Run dialog typeahead

Revision ID: 015
Revises: 014
Create Date: 2026-09-02

The Run dialog's symbol box was free text: a typo ran the whole pipeline
and failed late. This table is the local lookup the typeahead reads
instead of a vendor call per keystroke: S&P 500 and Russell 2000
constituents (source 'index', refreshed weekly), every symbol a run has
ever used (source 'run'), and unknown names that a single price lookup
accepted (source 'validated'). membership is a list of index tags such
as ['sp500'] or ['r2000'] so a cohort can be picked without a join.

Name search is a substring match; a plain btree on name keeps the
prefix case fast and needs no extension. A pg_trgm index would make
mid-word matches faster but the extension is not guaranteed on every
Postgres, so it is attempted and skipped rather than required.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tickers",
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("exchange", sa.String(length=16), nullable=True),
        sa.Column("membership", JSONB(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("symbol"),
    )
    op.create_index("ix_tickers_name", "tickers", ["name"], unique=False)
    # Optional: only when pg_trgm is already installed (CREATE EXTENSION
    # needs superuser on managed Postgres, so it is never attempted here).
    bind = op.get_bind()
    has_trgm = bind.execute(sa.text(
        "SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")).scalar()
    if has_trgm:
        op.execute("CREATE INDEX ix_tickers_name_trgm ON tickers "
                   "USING gin (name gin_trgm_ops)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_tickers_name_trgm")
    op.drop_index("ix_tickers_name", table_name="tickers")
    op.drop_table("tickers")
