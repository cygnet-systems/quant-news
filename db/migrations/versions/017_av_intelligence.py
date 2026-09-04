"""av intelligence: congressional trades, insider Form 4s, the Congress roster

Revision ID: 017
Revises: 016
Create Date: 2026-09-03

Reports built on price and indicators describe what already happened. These
tables hold the evidence a chart cannot show: who inside the company is
buying or selling, which member of Congress is trading the name, and when
either fact became public.

Both Alpha Vantage endpoints return the symbol's FULL history on every call
and ignore date parameters (NVDA: 408 congressional trades since 2016, 6,920
Form 4 lines since 2003), so they are not per-run queries. A symbol is
fetched once into these tables and every later run filters locally, which is
what makes the shared 70-calls/minute quota survivable across a watchlist.

visible_from is the correctness property. It is the date the disclosure
became public: the filing date for a congressional trade (notification date
where the vendor has no filing date, NULL and therefore never served when it
has neither), and for a Form 4 the second trading day after the transaction,
which is the SEC deadline and a documented proxy since the payload carries
no filing date at all. Every read filters visible_from <= as_of.

The uniqueness of a trade row is a digest column rather than a composite
constraint over the source fields: several of those fields are nullable, and
Postgres treats NULLs as distinct, so a composite UNIQUE would let the same
row re-insert on every weekly refresh.

This migration also installs the weekly 'av_refresh' job on a database that
already has a schedule, under the same guards 016 used for ticker_refresh:
seed_default_jobs only writes into an empty table, and a job that ran and was
then deleted (job_runs history survives the delete) must stay deleted.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JOB_ID = "av_refresh"

scheduled_jobs = sa.table(
    "scheduled_jobs",
    sa.column("id", sa.String),
    sa.column("kind", sa.String),
    sa.column("description", sa.Text),
    sa.column("enabled", sa.Boolean),
    sa.column("hour", sa.Integer),
    sa.column("minute", sa.Integer),
    sa.column("days_of_week", sa.String),
    sa.column("timezone", sa.String),
    sa.column("symbols_csv", sa.Text),
    sa.column("params_json", JSONB),
    sa.column("is_public", sa.Boolean),
)
job_runs = sa.table("job_runs", sa.column("job_id", sa.String))

# BIGINT on Postgres; SQLite needs a plain INTEGER for an autoincrementing
# primary key, and the migration is exercised on SQLite in tests.
_ROW_ID = sa.BigInteger().with_variant(sa.Integer, "sqlite")


def _install_job() -> None:
    bind = op.get_bind()
    total = bind.execute(
        sa.select(sa.func.count()).select_from(scheduled_jobs)).scalar()
    if not total:
        return
    has_kind = bind.execute(
        sa.select(sa.func.count()).select_from(scheduled_jobs)
        .where(scheduled_jobs.c.kind == JOB_ID)).scalar()
    has_history = bind.execute(
        sa.select(sa.func.count()).select_from(job_runs)
        .where(job_runs.c.job_id == JOB_ID)).scalar()
    if has_kind or has_history:
        return
    # Same spec as scheduler_service.DEFAULT_JOBS; tests pin the two together.
    bind.execute(scheduled_jobs.insert().values(
        id=JOB_ID,
        kind=JOB_ID,
        description="Refresh congressional trades, insider filings and the "
                    "Congress roster for the watchlist",
        enabled=True,
        hour=5,
        minute=30,
        days_of_week="sun",
        timezone="US/Eastern",
        symbols_csv=None,
        params_json={},
        is_public=True,
    ))


def upgrade() -> None:
    op.create_table(
        "politicians",
        sa.Column("bioguide_id", sa.String(length=16), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=True),
        sa.Column("chamber", sa.String(length=8), nullable=True),
        sa.Column("state", sa.String(length=4), nullable=True),
        sa.Column("district", sa.String(length=8), nullable=True),
        sa.Column("party", sa.String(length=4), nullable=True),
        sa.Column("aliases", JSONB(), nullable=True),
        sa.Column("terms", JSONB(), nullable=True),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("bioguide_id"),
    )
    op.create_index("ix_politicians_display_name", "politicians",
                    ["display_name"], unique=False)

    op.create_table(
        "politician_aliases",
        sa.Column("alias", sa.String(length=128), nullable=False),
        sa.Column("bioguide_id", sa.String(length=16), nullable=False),
        sa.PrimaryKeyConstraint("alias", "bioguide_id"),
    )
    op.create_index("ix_politician_aliases_alias", "politician_aliases",
                    ["alias"], unique=False)

    op.create_table(
        "congress_trades",
        sa.Column("id", _ROW_ID, autoincrement=True, nullable=False),
        sa.Column("natural_key", sa.String(length=40), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("bioguide_id", sa.String(length=16), nullable=True),
        sa.Column("politician_canonical", sa.Text(), nullable=True),
        sa.Column("chamber", sa.String(length=16), nullable=True),
        sa.Column("party", sa.String(length=8), nullable=True),
        sa.Column("state", sa.String(length=8), nullable=True),
        sa.Column("state_district", sa.String(length=16), nullable=True),
        sa.Column("asset_name", sa.Text(), nullable=True),
        sa.Column("asset_type_code", sa.String(length=8), nullable=True),
        sa.Column("transaction_type", sa.String(length=8), nullable=True),
        sa.Column("transaction_date", sa.Date(), nullable=True),
        sa.Column("notification_date", sa.Date(), nullable=True),
        sa.Column("filed_date", sa.Date(), nullable=True),
        sa.Column("amount_min", sa.Numeric(18, 2), nullable=True),
        sa.Column("amount_max", sa.Numeric(18, 2), nullable=True),
        sa.Column("owner_code", sa.String(length=16), nullable=True),
        sa.Column("filing_status", sa.String(length=16), nullable=True),
        # Nullable by design: a row with neither a filed nor a notification
        # date has no public date, and NULL <= as_of is never true, so the
        # read filter drops it without a special case.
        sa.Column("visible_from", sa.Date(), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("natural_key", name="uq_congress_trades_natural_key"),
    )
    op.create_index("ix_congress_trades_symbol", "congress_trades",
                    ["symbol", "visible_from"], unique=False)
    op.create_index("ix_congress_trades_member", "congress_trades",
                    ["bioguide_id", "visible_from"], unique=False)

    op.create_table(
        "insider_transactions",
        sa.Column("id", _ROW_ID, autoincrement=True, nullable=False),
        sa.Column("natural_key", sa.String(length=40), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("executive", sa.Text(), nullable=True),
        sa.Column("executive_title", sa.Text(), nullable=True),
        sa.Column("security_type", sa.Text(), nullable=True),
        sa.Column("acquisition_or_disposal", sa.String(length=2), nullable=True),
        sa.Column("transaction_date", sa.Date(), nullable=True),
        sa.Column("shares", sa.Numeric(20, 4), nullable=True),
        sa.Column("share_price", sa.Numeric(20, 4), nullable=True),
        # NULL when share_price is 0 or missing (grants, gifts, option
        # exercises): a computed dollar value there would be fiction.
        sa.Column("value_usd", sa.Numeric(24, 2), nullable=True),
        sa.Column("visible_from", sa.Date(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("natural_key",
                            name="uq_insider_transactions_natural_key"),
    )
    op.create_index("ix_insider_transactions_symbol", "insider_transactions",
                    ["symbol", "visible_from"], unique=False)

    op.create_table(
        "av_fetch_log",
        sa.Column("function", sa.String(length=32), nullable=False),
        sa.Column("subject", sa.String(length=32), nullable=False),
        sa.Column("last_fetched_at", sa.DateTime(timezone=True), nullable=True),
        # When a call was last ATTEMPTED and when data last actually arrived.
        # A block dates its rows from the second: reading the first would let
        # a failed top-up claim the store was synced today.
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rows_seen", sa.Integer(), nullable=True),
        sa.Column("ok", sa.Boolean(), server_default=sa.text("true"),
                  nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("function", "subject"),
    )

    _install_job()


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(scheduled_jobs.delete().where(scheduled_jobs.c.id == JOB_ID))
    op.drop_table("av_fetch_log")
    op.drop_index("ix_insider_transactions_symbol",
                  table_name="insider_transactions")
    op.drop_table("insider_transactions")
    op.drop_index("ix_congress_trades_member", table_name="congress_trades")
    op.drop_index("ix_congress_trades_symbol", table_name="congress_trades")
    op.drop_table("congress_trades")
    op.drop_index("ix_politician_aliases_alias", table_name="politician_aliases")
    op.drop_table("politician_aliases")
    op.drop_index("ix_politicians_display_name", table_name="politicians")
    op.drop_table("politicians")
