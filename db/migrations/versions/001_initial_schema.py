"""Initial schema: predictions, reports, recommendations, data snapshots.

Revision ID: 001
Revises: None
Create Date: 2026-07-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "data_snapshots",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("trade_date", sa.String(10), nullable=False),
        sa.Column("data_type", sa.String(32), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("record_count", sa.Integer),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "symbol", "trade_date", "data_type", name="uq_snapshot_key"
        ),
    )
    op.create_index(
        "ix_snapshot_lookup", "data_snapshots",
        ["symbol", "trade_date", "data_type"],
    )

    op.create_table(
        "prediction_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("trade_date", sa.String(10), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(32)),
        sa.Column("input_data_hash", sa.String(64), nullable=False),
        sa.Column("decision", sa.String(8), nullable=False),
        sa.Column("confidence", sa.Double),
        sa.Column("up_probability", sa.Double),
        sa.Column("result_json", JSONB),
        sa.Column("duration_ms", sa.Integer),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "symbol", "trade_date", "model_name", "input_data_hash",
            name="uq_prediction_cache",
        ),
    )
    op.create_index(
        "ix_prediction_lookup", "prediction_runs",
        ["symbol", "trade_date", "model_name"],
    )

    op.create_table(
        "report_catalog",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(16)),
        sa.Column("trade_date", sa.String(10), nullable=False),
        sa.Column("report_type", sa.String(64), nullable=False),
        sa.Column("input_data_hash", sa.String(64), nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=False, unique=True),
        sa.Column("file_format", sa.String(8), nullable=False, server_default="md"),
        sa.Column("size_bytes", sa.Integer),
        sa.Column("metadata_json", JSONB),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "symbol", "trade_date", "report_type", "input_data_hash",
            name="uq_report_cache",
        ),
    )
    op.create_index(
        "ix_report_lookup", "report_catalog",
        ["symbol", "trade_date", "report_type"],
    )

    op.create_table(
        "recommendation_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("trade_date", sa.String(10), nullable=False),
        sa.Column("symbols_csv", sa.Text, nullable=False),
        sa.Column("input_data_hash", sa.String(64), nullable=False),
        sa.Column("model_used", sa.String(64), nullable=False),
        sa.Column("provider_used", sa.String(32), nullable=False),
        sa.Column("result_json", JSONB),
        sa.Column("duration_ms", sa.Integer),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "trade_date", "input_data_hash", name="uq_recommendation_cache"
        ),
    )
    op.create_index(
        "ix_recommendation_lookup", "recommendation_runs", ["trade_date"]
    )


def downgrade() -> None:
    op.drop_table("recommendation_runs")
    op.drop_table("report_catalog")
    op.drop_table("prediction_runs")
    op.drop_table("data_snapshots")
