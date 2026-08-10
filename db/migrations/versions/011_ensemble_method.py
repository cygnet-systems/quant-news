"""Record which combination method produced an ensemble call.

The run dialog lets you pick between four ways of turning member votes into
one decision, but the choice was only ever visible inside details_json, and
only for runs after 2026-08-06. Measuring "does method matter" meant a JSONB
scan with no index and, in practice, one usable row out of 111.

Promoting it to a column makes hit-rate-by-method a normal grouped query.
Existing rows are backfilled from details_json where the key is present; the
rest stay NULL, meaning "not recorded" rather than any particular method.

Revision ID: 011
Revises: 010
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_predictions",
        sa.Column("ensemble_method", sa.String(length=32), nullable=True),
    )
    # Recover what history we have. Only ensemble rows carry the key, so this
    # leaves every other model NULL without needing a WHERE on model_name.
    op.execute("""
        UPDATE model_predictions
           SET ensemble_method = details_json->>'method'
         WHERE details_json ? 'method'
    """)
    # The question this exists to answer is "hit rate by method over a period".
    op.create_index(
        "ix_model_pred_ens_method",
        "model_predictions",
        ["ensemble_method", "prediction_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_pred_ens_method", table_name="model_predictions")
    op.drop_column("model_predictions", "ensemble_method")
