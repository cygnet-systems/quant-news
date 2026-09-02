"""ticker_refresh: install the weekly symbol-cache job on databases that
already have a schedule

Revision ID: 016
Revises: 015
Create Date: 2026-09-02

scheduler_service.seed_default_jobs only writes into an EMPTY
scheduled_jobs table, by design: seeding per id would bring back a job the
user deleted on the next restart. The cost of that rule is that a job kind
added after a database has its schedule (015 added the tickers table and
the weekly refresh that keeps it current) never appears there. This
migration is the one-time install for those databases; the seed still
covers a fresh one, so an empty table is left alone here or both paths
would race and the daily jobs would never be seeded.

Guards, so the row is written at most once and never resurrected: skip when
a job of this kind already exists, and skip when job_runs holds history for
the id (delete_job keeps the runs, so a job that ran and was then deleted
leaves exactly that trace).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JOB_ID = "ticker_refresh"

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


def upgrade() -> None:
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
        description="Refresh the symbol lookup cache from the index lists",
        enabled=True,
        hour=6,
        minute=0,
        days_of_week="sun",
        timezone="US/Eastern",
        symbols_csv=None,
        params_json={},
        is_public=True,
    ))


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(scheduled_jobs.delete().where(scheduled_jobs.c.id == JOB_ID))
