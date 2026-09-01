"""Per-run trace capture: llm_traces bodies + run_id joins + event payloads.

Three gaps this closes, together forming the Trace view's storage:

* No prompt text was stored anywhere — llm_usage records tokens/cost per call
  but the exact system prompt, prompt and raw response were discarded after
  parsing, so "why did it conclude this?" had no answer. llm_traces keeps the
  bodies, one row per PHYSICAL API call (retries and failover attempts
  included); llm_usage remains the sole cost record.
* model_predictions (and the report/recommendation tables) had no run_id, so
  a run could not be joined to what it produced.
* activity_log events were prose only; the payload column carries the
  structured facts (counts, windows, hashes) the message summarises.

Revision ID: 012
Revises: 011
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_traces",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        # Row id of the paired llm_usage record (tokens/cost live there).
        sa.Column("usage_id", sa.Integer(), nullable=True),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=False,
                  server_default="unknown"),
        sa.Column("section", sa.String(length=64), nullable=True),
        sa.Column("symbol", sa.String(length=16), nullable=True),
        sa.Column("trade_date", sa.Date(), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("response", sa.Text(), nullable=True),
        sa.Column("params_json", JSONB(), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False,
                  server_default=sa.text("1")),
        sa.Column("ok", sa.Boolean(), nullable=False,
                  server_default=sa.text("true")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("owner_uid", sa.String(length=64), nullable=True),
        sa.Column("is_public", sa.Boolean(), nullable=False,
                  server_default=sa.text("true")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_traces_run", "llm_traces", ["run_id"])
    op.create_index("ix_llm_traces_time", "llm_traces", ["created_at"])
    op.create_index("ix_llm_traces_owner", "llm_traces", ["owner_uid"])

    op.add_column("activity_log", sa.Column("payload", JSONB(), nullable=True))

    # run_id on everything a run produces, so the Trace page can join a run
    # to its outputs. NULL means "written before runs were stamped".
    op.add_column("model_predictions",
                  sa.Column("run_id", sa.String(length=36), nullable=True))
    op.create_index("ix_model_pred_run", "model_predictions", ["run_id"])
    op.add_column("trading_agent_reports",
                  sa.Column("run_id", sa.String(length=36), nullable=True))
    op.create_index("ix_trading_agent_report_run", "trading_agent_reports",
                    ["run_id"])
    op.add_column("recommendation_runs",
                  sa.Column("run_id", sa.String(length=36), nullable=True))
    op.create_index("ix_recommendation_run", "recommendation_runs", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_recommendation_run", table_name="recommendation_runs")
    op.drop_column("recommendation_runs", "run_id")
    op.drop_index("ix_trading_agent_report_run",
                  table_name="trading_agent_reports")
    op.drop_column("trading_agent_reports", "run_id")
    op.drop_index("ix_model_pred_run", table_name="model_predictions")
    op.drop_column("model_predictions", "run_id")
    op.drop_column("activity_log", "payload")
    op.drop_index("ix_llm_traces_owner", table_name="llm_traces")
    op.drop_index("ix_llm_traces_time", table_name="llm_traces")
    op.drop_index("ix_llm_traces_run", table_name="llm_traces")
    op.drop_table("llm_traces")
