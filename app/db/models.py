"""SQLAlchemy ORM model for the audit store.

Cross-dialect on purpose so the same metadata drives Postgres (prod) and SQLite
(the Alembic reversibility check): `Uuid` -> native UUID on PG / CHAR on others,
and JSON `with_variant(JSONB)` -> JSONB on PG (indexable) / JSON elsewhere.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on Postgres, plain JSON everywhere else.
_JSON = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class ClaimDecision(Base):
    __tablename__ = "claim_decisions"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True)
    request_id: Mapped[str] = mapped_column(String, nullable=False)
    correlation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String, nullable=False)
    endpoint: Mapped[str] = mapped_column(String, nullable=False)
    order_id: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    image_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    image_phash: Mapped[str | None] = mapped_column(String, nullable=True)

    model_name: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_version: Mapped[str] = mapped_column(String, nullable=False)

    raw_observations: Mapped[dict] = mapped_column(_JSON, nullable=False, default=dict)
    computed: Mapped[dict] = mapped_column(_JSON, nullable=False, default=dict)
    response_snapshot: Mapped[dict] = mapped_column(_JSON, nullable=False, default=dict)

    final_action: Mapped[str] = mapped_column(String, nullable=False)
    final_status: Mapped[str | None] = mapped_column(String, nullable=True)
    priority: Mapped[str | None] = mapped_column(String, nullable=True)
    routed_to: Mapped[str] = mapped_column(String, nullable=False)

    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ---- Dispute verification (/dispute) — nullable, expand/contract safe -----
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    category_source: Mapped[str | None] = mapped_column(String, nullable=True)
    decision: Mapped[str | None] = mapped_column(String, nullable=True)
    route: Mapped[str | None] = mapped_column(String, nullable=True)
    agent_flags: Mapped[dict | None] = mapped_column(_JSON, nullable=True)
    refund: Mapped[dict | None] = mapped_column(_JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_claim_decisions_idem"),
        CheckConstraint(
            "endpoint IN ('scan','verify_claim','dispute')",
            name="ck_claim_decisions_endpoint",
        ),
        CheckConstraint(
            "routed_to IN ('auto','human','challenge')", name="ck_claim_decisions_routed"
        ),
        Index("ix_claim_decisions_order_id", "order_id"),
        Index("ix_claim_decisions_user_created", "user_id", "created_at"),
        Index("ix_claim_decisions_phash", "image_phash"),
    )
