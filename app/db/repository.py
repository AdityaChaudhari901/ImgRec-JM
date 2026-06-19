"""Audit-store repository: the only layer that knows about persistence.

`DecisionRecord` is the plain domain object the service layer builds; the repo
maps it to/from storage. Two implementations, selected by `DATABASE_URL`:

  - InMemoryDecisionRepository : dev/tests, no DB driver needed.
  - SqlAlchemyDecisionRepository: Postgres (imported lazily — see get_decision_repository).

Idempotency is enforced at the store: `insert` raises `DuplicateDecision` when the
idempotency key already exists, so the service can fetch and replay the winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DuplicateDecision(Exception):
    """Raised by insert() when the idempotency key is already persisted."""


@dataclass
class DecisionRecord:
    """One immutable, money-affecting decision (the audit row)."""

    id: str
    request_id: str
    idempotency_key: str
    endpoint: str
    order_id: str
    user_id: str
    model_name: str
    prompt_version: str
    final_action: str
    routed_to: str
    response_snapshot: dict  # the exact response returned to Kaily (verbatim replay)
    correlation_id: Optional[str] = None
    image_ref: Optional[str] = None
    image_phash: Optional[str] = None
    model_version: Optional[str] = None
    final_status: Optional[str] = None
    priority: Optional[str] = None
    raw_observations: dict = field(default_factory=dict)
    computed: dict = field(default_factory=dict)
    latency_ms: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionRepository(Protocol):
    async def get_by_idempotency_key(self, key: str) -> Optional[DecisionRecord]:
        ...

    async def insert(self, record: DecisionRecord) -> None:
        """Persist a record. Raises DuplicateDecision if the key already exists."""
        ...


class InMemoryDecisionRepository:
    """Process-local audit store for dev/tests. NOT durable — prod fails fast if
    DATABASE_URL is unset (see settings)."""

    def __init__(self) -> None:
        self._by_key: dict[str, DecisionRecord] = {}

    async def get_by_idempotency_key(self, key: str) -> Optional[DecisionRecord]:
        return self._by_key.get(key)

    async def insert(self, record: DecisionRecord) -> None:
        if record.idempotency_key in self._by_key:
            raise DuplicateDecision(record.idempotency_key)
        self._by_key[record.idempotency_key] = record

    def reset(self) -> None:
        self._by_key.clear()


_repo: Optional[DecisionRepository] = None


def get_decision_repository() -> DecisionRepository:
    """Return the configured repository (cached singleton).

    Postgres when DATABASE_URL is set; otherwise the in-memory fallback with a
    loud warning (dev only — production refuses to boot without a real DSN).
    """
    global _repo
    if _repo is None:
        if settings.database_url:
            from app.db.sql_repository import SqlAlchemyDecisionRepository  # lazy

            _repo = SqlAlchemyDecisionRepository()
        else:
            logger.warning(
                "audit_store_in_memory",
                detail="DATABASE_URL unset — using non-durable in-memory audit store",
            )
            _repo = InMemoryDecisionRepository()
    return _repo


def reset_decision_repository() -> None:
    """Test hook — drop the singleton so each test starts clean."""
    global _repo
    if isinstance(_repo, InMemoryDecisionRepository):
        _repo.reset()
    _repo = None
