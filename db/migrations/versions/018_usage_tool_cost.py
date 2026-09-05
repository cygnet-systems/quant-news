"""Record what a server-side web search cost, not just the tokens around it.

llm_usage priced tokens and nothing else, so the investigation stage -- the
one stage that pays a provider to browse -- reported roughly half its true
cost. On 2026-09-04 the table said $9.67 for the day against a real bill near
$20; the difference was 881 billed web searches that nothing recorded.

``searches`` is an observation: the count the provider's response carried.
``tool_cost_usd`` is that count priced by config.WEB_SEARCH_PRICING, and is
NULL when the rate is unknown, the same way an unpriced model records a NULL
cost rather than a guessed zero. ``cost_usd`` keeps its meaning (tokens), so
every historical row still says what it always said; readers that want the
real number add the two.

``cached_input_tokens`` is the provider-reported share of the input it served
from its prompt cache. Without it nobody can tell a prompt that caches from
one that cannot, which is how "add prompt caching" stays an assumption
instead of a measurement.

Revision ID: 018
Revises: 017
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_usage", sa.Column(
        "searches", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("llm_usage", sa.Column(
        "tool_cost_usd", sa.Double(), nullable=True))
    op.add_column("llm_usage", sa.Column(
        "cached_input_tokens", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("llm_usage", "cached_input_tokens")
    op.drop_column("llm_usage", "tool_cost_usd")
    op.drop_column("llm_usage", "searches")
