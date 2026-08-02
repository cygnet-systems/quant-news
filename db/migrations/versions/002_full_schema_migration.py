"""Full schema migration — DuckDB tables to Postgres, merge prediction_runs into model_predictions.

Revision ID: 002
Revises: 001
Create Date: 2026-07-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Drop prediction_runs (replaced by model_predictions) ──
    op.drop_index("ix_prediction_lookup", table_name="prediction_runs")
    op.drop_table("prediction_runs")

    # ── stock_prices ──
    op.create_table(
        "stock_prices",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("open", sa.Double),
        sa.Column("high", sa.Double),
        sa.Column("low", sa.Double),
        sa.Column("close", sa.Double),
        sa.Column("volume", sa.BigInteger),
        sa.Column("dividends", sa.Double),
        sa.Column("stock_splits", sa.Double),
        sa.Column("fetched_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("symbol", "date", name="uq_stock_price_key"),
    )
    op.create_index(
        "ix_stock_price_lookup", "stock_prices", ["symbol", "date"]
    )

    # ── stock_info ──
    op.create_table(
        "stock_info",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(16), nullable=False, unique=True),
        sa.Column("name", sa.String),
        sa.Column("sector", sa.String),
        sa.Column("industry", sa.String),
        sa.Column("market_cap", sa.BigInteger),
        sa.Column("current_price", sa.Double),
        sa.Column("previous_close", sa.Double),
        sa.Column("pe_ratio", sa.Double),
        sa.Column("dividend_yield", sa.Double),
        sa.Column("fifty_two_week_high", sa.Double),
        sa.Column("fifty_two_week_low", sa.Double),
        sa.Column("volume", sa.BigInteger),
        sa.Column("avg_volume", sa.BigInteger),
        sa.Column("fetched_at", sa.DateTime(timezone=True)),
    )

    # ── cache_metadata ──
    op.create_table(
        "cache_metadata",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("data_type", sa.String(32), nullable=False),
        sa.Column("period", sa.String(8)),
        sa.Column("last_updated", sa.DateTime(timezone=True)),
        sa.Column("record_count", sa.Integer),
        sa.UniqueConstraint("symbol", "data_type", name="uq_cache_metadata_key"),
    )

    # ── news_articles ──
    op.create_table(
        "news_articles",
        sa.Column("id", sa.String, primary_key=True, autoincrement=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("title", sa.Text),
        sa.Column("source", sa.Text),
        sa.Column("url", sa.Text),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("summary", sa.Text),
        sa.Column("sentiment", sa.String(16)),
        sa.Column("sentiment_score", sa.Double),
        sa.Column("fetched_at", sa.DateTime(timezone=True)),
        sa.Column("topics_json", sa.Text),
        sa.Column("overall_sentiment_score", sa.Double),
        sa.Column("overall_sentiment_label", sa.String(32)),
        sa.Column("ticker_relevance_score", sa.Double),
        sa.Column("impact", sa.String(16)),
    )
    op.create_index(
        "ix_news_article_lookup", "news_articles", ["symbol", "fetched_at"]
    )

    # ── model_predictions ──
    op.create_table(
        "model_predictions",
        sa.Column("id", sa.String, primary_key=True, autoincrement=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("prediction_date", sa.Date, nullable=False),
        sa.Column("target_date", sa.Date, nullable=False),
        sa.Column("decision", sa.String(8), nullable=False),
        sa.Column("confidence", sa.Double),
        sa.Column("up_probability", sa.Double),
        sa.Column("predicted_close", sa.Double),
        sa.Column("previous_close", sa.Double),
        sa.Column("actual_close", sa.Double),
        sa.Column("was_correct", sa.Boolean),
        sa.Column("pnl_dollars", sa.Double),
        sa.Column("model_version", sa.String(32)),
        sa.Column("training_samples", sa.Integer),
        sa.Column("feature_values_json", sa.Text),
        sa.Column("details_json", JSONB),
        sa.Column("input_data_hash", sa.String(64)),
        sa.Column("duration_ms", sa.Integer),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
        sa.Column("evaluated_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_model_pred_sym_model_date", "model_predictions",
        ["symbol", "model_name", "prediction_date"],
    )
    op.create_index(
        "ix_model_pred_sym_date", "model_predictions",
        ["symbol", "prediction_date"],
    )
    op.create_index(
        "ix_model_pred_hash", "model_predictions", ["input_data_hash"]
    )

    # ── historical_news ──
    op.create_table(
        "historical_news",
        sa.Column("id", sa.String, primary_key=True, autoincrement=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("published_date", sa.Date),
        sa.Column("title", sa.Text),
        sa.Column("summary", sa.Text),
        sa.Column("url", sa.Text),
        sa.Column("source", sa.Text),
        sa.Column("topics_json", sa.Text),
        sa.Column("overall_sentiment_score", sa.Double),
        sa.Column("overall_sentiment_label", sa.String(32)),
        sa.Column("ticker_sentiment_score", sa.Double),
        sa.Column("ticker_relevance_score", sa.Double),
        sa.Column("fetched_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_historical_news_lookup", "historical_news",
        ["symbol", "published_date"],
    )

    # ── strategy_evaluations ──
    op.create_table(
        "strategy_evaluations",
        sa.Column("id", sa.String, primary_key=True, autoincrement=False),
        sa.Column(
            "prediction_id", sa.String,
            sa.ForeignKey("model_predictions.id"), nullable=False,
        ),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("strategy_version", sa.String(32)),
        sa.Column("action", sa.String(8), nullable=False),
        sa.Column("position_size", sa.Double, server_default="1000.0"),
        sa.Column("entry_price", sa.Double),
        sa.Column("exit_price", sa.Double),
        sa.Column("pnl_dollars", sa.Double),
        sa.Column("was_correct", sa.Boolean),
        sa.Column("signal_metadata_json", sa.Text),
        sa.Column(
            "evaluated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_strategy_eval_lookup", "strategy_evaluations",
        ["strategy_name", "prediction_id"],
    )

    # ── strategy_metrics ──
    op.create_table(
        "strategy_metrics",
        sa.Column("id", sa.String, primary_key=True, autoincrement=False),
        sa.Column("strategy_name", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(16)),
        sa.Column("period", sa.String(16), nullable=False),
        sa.Column("sharpe_ratio", sa.Double),
        sa.Column("sortino_ratio", sa.Double),
        sa.Column("max_drawdown", sa.Double),
        sa.Column("win_rate", sa.Double),
        sa.Column("total_pnl", sa.Double),
        sa.Column("total_trades", sa.Integer),
        sa.Column("metrics_json", JSONB),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_strategy_metrics_lookup", "strategy_metrics",
        ["strategy_name", "symbol"],
    )

    # ── trading_agent_reports ──
    op.create_table(
        "trading_agent_reports",
        sa.Column("id", sa.String, primary_key=True, autoincrement=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("trade_date", sa.Date, nullable=False),
        sa.Column("decision", sa.String(8), nullable=False),
        sa.Column("confidence", sa.Double),
        sa.Column("report_text", sa.Text),
        sa.Column("model_name", sa.String(64)),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("output_tokens", sa.Integer),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_trading_agent_report_lookup", "trading_agent_reports",
        ["symbol", "trade_date"],
    )


def downgrade() -> None:
    # Drop new tables in reverse dependency order
    op.drop_index("ix_trading_agent_report_lookup", table_name="trading_agent_reports")
    op.drop_table("trading_agent_reports")

    op.drop_index("ix_strategy_metrics_lookup", table_name="strategy_metrics")
    op.drop_table("strategy_metrics")

    op.drop_index("ix_strategy_eval_lookup", table_name="strategy_evaluations")
    op.drop_table("strategy_evaluations")

    op.drop_index("ix_historical_news_lookup", table_name="historical_news")
    op.drop_table("historical_news")

    op.drop_index("ix_model_pred_hash", table_name="model_predictions")
    op.drop_index("ix_model_pred_sym_date", table_name="model_predictions")
    op.drop_index("ix_model_pred_sym_model_date", table_name="model_predictions")
    op.drop_table("model_predictions")

    op.drop_index("ix_news_article_lookup", table_name="news_articles")
    op.drop_table("news_articles")

    op.drop_table("cache_metadata")

    op.drop_table("stock_info")

    op.drop_index("ix_stock_price_lookup", table_name="stock_prices")
    op.drop_table("stock_prices")

    # Recreate prediction_runs (from 001)
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
