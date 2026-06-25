"""add dispute columns + widen endpoint check constraint

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-25

Expand/contract: all new columns are nullable, so the old app version keeps
working during the rollout. The endpoint CHECK constraint is widened to admit
'dispute'. Reversible: downgrade() restores the prior constraint and drops the
columns.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("claim_decisions", sa.Column("category", sa.String(), nullable=True))
    op.add_column("claim_decisions", sa.Column("category_source", sa.String(), nullable=True))
    op.add_column("claim_decisions", sa.Column("decision", sa.String(), nullable=True))
    op.add_column("claim_decisions", sa.Column("route", sa.String(), nullable=True))
    op.add_column("claim_decisions", sa.Column("agent_flags", _JSON, nullable=True))
    op.add_column("claim_decisions", sa.Column("refund", _JSON, nullable=True))

    # Widen the endpoint CHECK constraint to admit 'dispute'. batch_alter_table
    # keeps this reversible/cross-dialect (Postgres prod, SQLite local check).
    with op.batch_alter_table("claim_decisions") as batch:
        batch.drop_constraint("ck_claim_decisions_endpoint", type_="check")
        batch.create_check_constraint(
            "ck_claim_decisions_endpoint", "endpoint IN ('scan','verify_claim','dispute')"
        )


def downgrade() -> None:
    with op.batch_alter_table("claim_decisions") as batch:
        batch.drop_constraint("ck_claim_decisions_endpoint", type_="check")
        batch.create_check_constraint(
            "ck_claim_decisions_endpoint", "endpoint IN ('scan','verify_claim')"
        )
    op.drop_column("claim_decisions", "refund")
    op.drop_column("claim_decisions", "agent_flags")
    op.drop_column("claim_decisions", "route")
    op.drop_column("claim_decisions", "decision")
    op.drop_column("claim_decisions", "category_source")
    op.drop_column("claim_decisions", "category")
