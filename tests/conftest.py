"""Test bootstrap.

Set required configuration BEFORE `app` (and therefore `settings`) is imported
by any test module. The shared secret is pinned to `test-secret` so the value in
the tests' `x-api-key` header authenticates successfully, while a wrong key still
returns 401. No real Google credentials are needed — the Gemini boundary is
always mocked.
"""

import os

os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("VERTEX_PROJECT_ID", "test-project")
os.environ.setdefault("KAILY_API_SECRET", "test-secret")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.0-flash-001")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import pytest

from app.db.repository import reset_decision_repository
from app.services.dedup_index import reset_dedup_index
from app.storage.object_store import reset_object_store


@pytest.fixture(autouse=True)
def _reset_audit_state():
    """Isolate the in-memory audit store, object store, and dedup index between
    tests, so state from one test doesn't leak into the next (false replays/dups)."""
    reset_decision_repository()
    reset_object_store()
    reset_dedup_index()
    yield
    reset_decision_repository()
    reset_object_store()
    reset_dedup_index()
