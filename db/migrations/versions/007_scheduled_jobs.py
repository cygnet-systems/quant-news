"""Add scheduled_jobs + job_runs.

Moves scheduling into the application. The app already ran an in-process
APScheduler for the daily evaluation with the schedule hardcoded in config;
this makes the schedule data. Editable from the dashboard, surviving a
redeploy: and adds a run history so a missed or failed run is visible rather
than inferred from logs.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scheduled_jobs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False,
                  server_default=sa.text("true")),
        sa.Column("hour", sa.Integer(), nullable=False, server_default="8"),
        sa.Column("minute", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("days_of_week", sa.String(length=32), nullable=False,
                  server_default="mon-fri"),
        sa.Column("timezone", sa.String(length=64), nullable=False,
                  server_default="US/Eastern"),
        sa.Column("symbols_csv", sa.Text(), nullable=True),
        sa.Column("params_json", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status", sa.String(length=16), nullable=True),
        sa.Column("last_detail", sa.Text(), nullable=True),
        sa.Column("last_duration_ms", sa.Integer(), nullable=True),
        sa.Column("last_success_date", sa.String(length=10), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("owner_uid", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "job_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("trigger", sa.String(length=16), nullable=False,
                  server_default="schedule"),
        sa.Column("status", sa.String(length=16), nullable=False,
                  server_default="running"),
        sa.Column("started_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("owner_uid", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_job_runs_job_time", "job_runs", ["job_id", "started_at"])


def downgrade() -> None:
    op.drop_index("ix_job_runs_job_time", table_name="job_runs")
    op.drop_table("job_runs")
    op.drop_table("scheduled_jobs")
