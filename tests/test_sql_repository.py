"""Integration test for the Postgres-backed repository.

Per backend-architecture, the repository layer is tested against a real SQL engine
(not a mock). We use async SQLite (aiosqlite) so it runs in CI with no Postgres
server — the SQLAlchemy code path (sessions, commit, IntegrityError -> DuplicateDecision)
is identical. Skipped if SQLAlchemy/aiosqlite aren't installed.
"""

import contextlib

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.db import engine as engine_mod  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.db.repository import DecisionRecord, DuplicateDecision  # noqa: E402
from app.db.sql_repository import SqlAlchemyDecisionRepository  # noqa: E402


@contextlib.asynccontextmanager
async def _sql_repo(tmp_path):
    """Point app.db.engine at a throwaway async-SQLite DB for the duration."""
    url = f"sqlite+aiosqlite:///{tmp_path}/audit.db"
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    saved_engine, saved_maker = engine_mod._engine, engine_mod._sessionmaker
    engine_mod._engine = eng
    engine_mod._sessionmaker = async_sessionmaker(eng, expire_on_commit=False)
    try:
        yield SqlAlchemyDecisionRepository()
    finally:
        await eng.dispose()
        engine_mod._engine, engine_mod._sessionmaker = saved_engine, saved_maker


def _record(key: str = "k1") -> DecisionRecord:
    return DecisionRecord(
        id="11111111-1111-1111-1111-111111111111",
        request_id="req_1",
        idempotency_key=key,
        endpoint="scan",
        order_id="JM-1",
        user_id="u_1",
        model_name="gemini-2.0-flash-001",
        prompt_version="scan-v1",
        final_action="initiate_refund",
        routed_to="auto",
        response_snapshot={"request_id": "req_1", "status": "expired"},
        image_ref="scan/2026/06/18/abc.jpg",
        image_phash="deadbeefdeadbeef",
        raw_observations={"status": "expired"},
        computed={"days_since_expiry": 28},
        latency_ms=120,
    )


async def test_insert_then_get_roundtrips(tmp_path):
    async with _sql_repo(tmp_path) as repo:
        await repo.insert(_record("rt"))
        got = await repo.get_by_idempotency_key("rt")
    assert got is not None
    assert got.endpoint == "scan"
    assert got.final_action == "initiate_refund"
    assert got.computed["days_since_expiry"] == 28
    assert got.response_snapshot["status"] == "expired"


async def test_get_missing_returns_none(tmp_path):
    async with _sql_repo(tmp_path) as repo:
        assert await repo.get_by_idempotency_key("nope") is None


async def test_duplicate_idempotency_key_raises(tmp_path):
    async with _sql_repo(tmp_path) as repo:
        await repo.insert(_record("dup"))
        with pytest.raises(DuplicateDecision):
            await repo.insert(_record("dup"))
