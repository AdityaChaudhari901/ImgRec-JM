"""Postgres-backed DecisionRepository (imported only when DATABASE_URL is set)."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.engine import get_sessionmaker
from app.db.models import ClaimDecision
from app.db.repository import DecisionRecord, DuplicateDecision

_COLUMNS = (
    "id",
    "request_id",
    "correlation_id",
    "idempotency_key",
    "endpoint",
    "order_id",
    "user_id",
    "created_at",
    "image_ref",
    "image_phash",
    "model_name",
    "model_version",
    "prompt_version",
    "raw_observations",
    "computed",
    "response_snapshot",
    "final_action",
    "final_status",
    "priority",
    "routed_to",
    "latency_ms",
)


def _to_orm(record: DecisionRecord) -> ClaimDecision:
    return ClaimDecision(**{name: getattr(record, name) for name in _COLUMNS})


def _to_record(row: ClaimDecision) -> DecisionRecord:
    return DecisionRecord(**{name: getattr(row, name) for name in _COLUMNS})


class SqlAlchemyDecisionRepository:
    async def get_by_idempotency_key(self, key: str) -> Optional[DecisionRecord]:
        async with get_sessionmaker()() as session:
            result = await session.execute(
                select(ClaimDecision).where(ClaimDecision.idempotency_key == key)
            )
            row = result.scalar_one_or_none()
            return _to_record(row) if row is not None else None

    async def insert(self, record: DecisionRecord) -> None:
        async with get_sessionmaker()() as session:
            session.add(_to_orm(record))
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise DuplicateDecision(record.idempotency_key) from exc
