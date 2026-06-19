"""create claim_decisions audit table

Revision ID: 0001
Revises:
Create Date: 2026-06-18

Reversible by design: upgrade() creates the table + indexes + constraints;
downgrade() drops them. Non-destructive (new table only, no data backfill).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# JSONB on Postgres, generic JSON on other dialects (e.g. SQLite for the local check).
_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "claim_decisions",
        sa.Column("id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("correlation_id", sa.String(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=False),
        sa.Column("endpoint", sa.String(), nullable=False),
        sa.Column("order_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("image_ref", sa.String(), nullable=True),
        sa.Column("image_phash", sa.String(), nullable=True),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("prompt_version", sa.String(), nullable=False),
        sa.Column("raw_observations", _JSON, nullable=False),
        sa.Column("computed", _JSON, nullable=False),
        sa.Column("response_snapshot", _JSON, nullable=False),
        sa.Column("final_action", sa.String(), nullable=False),
        sa.Column("final_status", sa.String(), nullable=True),
        sa.Column("priority", sa.String(), nullable=True),
        sa.Column("routed_to", sa.String(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.UniqueConstraint("idempotency_key", name="uq_claim_decisions_idem"),
        sa.CheckConstraint(
            "endpoint IN ('scan','verify_claim')", name="ck_claim_decisions_endpoint"
        ),
        sa.CheckConstraint(
            "routed_to IN ('auto','human','challenge')", name="ck_claim_decisions_routed"
        ),
    )
    op.create_index("ix_claim_decisions_order_id", "claim_decisions", ["order_id"])
    op.create_index(
        "ix_claim_decisions_user_created", "claim_decisions", ["user_id", "created_at"]
    )
    op.create_index("ix_claim_decisions_phash", "claim_decisions", ["image_phash"])


def downgrade() -> None:
    op.drop_index("ix_claim_decisions_phash", table_name="claim_decisions")
    op.drop_index("ix_claim_decisions_user_created", table_name="claim_decisions")
    op.drop_index("ix_claim_decisions_order_id", table_name="claim_decisions")
    op.drop_table("claim_decisions")
