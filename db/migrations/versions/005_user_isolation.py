"""Add owner_uid/is_public to user-generated content tables.

Groundwork for per-user data isolation behind Cygnet SSO. Everything existing
and everything written anonymously stays public (is_public defaults TRUE,
owner_uid NULL) — behavior is unchanged until a signed-in user creates
private content.

The auth `users`/`sessions` tables are NOT managed here: they mirror
CygnetResearchTerminal's and are created by services.auth_service on startup
(or already exist in the shared AUTH_DATABASE_URL database).

Revision ID: 005
Revises: 092c96e37b6d
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "092c96e37b6d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = (
    ("model_predictions", "ix_model_pred_owner"),
    ("trading_agent_reports", "ix_trading_agent_report_owner"),
    ("recommendation_runs", "ix_recommendation_owner"),
    ("report_catalog", "ix_report_owner"),
)


def upgrade() -> None:
    for table, index in _TABLES:
        op.add_column(table, sa.Column("owner_uid", sa.String(64), nullable=True))
        op.add_column(
            table,
            sa.Column("is_public", sa.Boolean(), nullable=False,
                      server_default=sa.text("true")),
        )
        op.create_index(index, table, ["owner_uid"])


def downgrade() -> None:
    for table, index in _TABLES:
        op.drop_index(index, table_name=table)
        op.drop_column(table, "is_public")
        op.drop_column(table, "owner_uid")
