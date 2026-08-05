"""Add llm_usage telemetry table.

One row per LLM API call: exact token counts from the provider response, the
$/Mtok rates applied, and the derived cost. Rates are stored per row on
purpose — repricing a model must not rewrite the cost of calls already made.

Nothing in this table is ever read back into a prompt; it exists to attribute
spend to the stage that caused it (research / ai_report / recommendations).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_usage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=False,
                  server_default="unknown"),
        sa.Column("symbol", sa.String(length=16), nullable=True),
        sa.Column("trade_date", sa.String(length=10), nullable=True),
        sa.Column("provider", sa.String(length=32), nullable=True),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_rate_per_mtok", sa.Double(), nullable=True),
        sa.Column("output_rate_per_mtok", sa.Double(), nullable=True),
        sa.Column("cost_usd", sa.Double(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("ok", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("owner_uid", sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_usage_time", "llm_usage", ["created_at"])
    op.create_index("ix_llm_usage_stage", "llm_usage", ["stage", "created_at"])
    op.create_index("ix_llm_usage_run", "llm_usage", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_llm_usage_run", table_name="llm_usage")
    op.drop_index("ix_llm_usage_stage", table_name="llm_usage")
    op.drop_index("ix_llm_usage_time", table_name="llm_usage")
    op.drop_table("llm_usage")
