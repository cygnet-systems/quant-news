"""analysis_runs: one row per analysis run, manual or scheduled

Revision ID: 014
Revises: 013
Create Date: 2026-09-02

Until now only scheduled jobs left a record (job_runs); a run started from
the Run dialog existed only as free-text lines on a single global progress
channel, so there was nothing to key a run page, a per-owner lock, a cancel,
or a completion link on, and two users running at once shared one run_id.
This table is the run record: status lifecycle (queued, running, done,
failed, cancelled), the symbols and config it was started with, per-stage
progress and counters kept as JSON so the runner can report whatever it
knows without a schema change, and job_run_id linking a scheduled run to
the job_runs row that launched it.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_runs",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("owner_uid", sa.String(length=64), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("preset", sa.String(length=16), nullable=True),
        sa.Column("config_json", JSONB(), nullable=True),
        sa.Column("symbols_csv", sa.Text(), nullable=False),
        sa.Column("prediction_date", sa.Date(), nullable=True),
        sa.Column("target_date", sa.Date(), nullable=True),
        sa.Column("stages_json", JSONB(), nullable=True),
        sa.Column("counters_json", JSONB(), nullable=True),
        sa.Column("estimate_s", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("job_run_id", sa.Integer(), nullable=True),
        sa.Column("is_public", sa.Boolean(), server_default=sa.text("true"),
                  nullable=False),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_analysis_runs_owner_status", "analysis_runs",
                    ["owner_uid", "status"], unique=False)
    op.create_index("ix_analysis_runs_started", "analysis_runs",
                    ["started_at"], unique=False)
    op.create_index("ix_analysis_runs_kind_date", "analysis_runs",
                    ["kind", "prediction_date"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_analysis_runs_kind_date", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_started", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_owner_status", table_name="analysis_runs")
    op.drop_table("analysis_runs")
