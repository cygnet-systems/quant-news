"""Record whether a prediction's news actually arrived.

Alpha Vantage answers an over-quota request with HTTP 200 and an explanatory
body. Before that was detected, a throttled fetch reached the models as an
empty article list, so a prediction made with no news at all was stored and
scored exactly like one made on a full window. TYL on 2026-08-04 is the worked
example: 48 articles existed, the models saw none, and the research report
presented the hole as a finding.

The runner now separates "the source failed" from "the window was quiet", but
that distinction only lived in the run log. This column carries it onto the
prediction, so the scoreboard can exclude unsupported calls instead of
averaging them in.

Existing rows stay NULL. NULL means "made before this was recorded", which is
deliberately not the same as "ok" - backfilling them to "ok" would assert
something we cannot know.

Revision ID: 010
Revises: 009
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model_predictions",
        sa.Column("news_status", sa.String(length=16), nullable=True),
    )
    # The scoreboard's "exclude unsupported calls" filter scans by status over
    # a date range, so index the pair rather than the status alone.
    op.create_index(
        "ix_model_pred_news_status",
        "model_predictions",
        ["news_status", "prediction_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_pred_news_status", table_name="model_predictions")
    op.drop_column("model_predictions", "news_status")
