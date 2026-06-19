"""Pluggable object storage for uploaded claim images.

The DB stores only a *reference* (the returned key) + perceptual hash — never the
base64 blob (Phase 1 principle: no images in Postgres). Providers are selected by
`OBJECT_STORE_PROVIDER`, mirroring the pluggable `ai_detector_provider` pattern:

  - "memory" : in-process dict — dev/tests, zero side effects (default).
  - "local"  : filesystem under OBJECT_STORE_LOCAL_DIR — local demos.
  - "gcs"    : Google Cloud Storage — production (India region per DPDP).

`put` returns the storage key persisted on the audit row. The heavy GCS SDK is
imported lazily so the test suite never needs it.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}


def _make_key(content_type: str, prefix: str) -> str:
    """`<prefix>/<YYYY>/<MM>/<DD>/<uuid>.<ext>` — sortable, collision-free."""
    ext = _EXT.get(content_type, "bin")
    day = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    head = f"{prefix.strip('/')}/" if prefix else ""
    return f"{head}{day}/{uuid.uuid4().hex}.{ext}"


class ObjectStore(Protocol):
    async def put(self, data: bytes, content_type: str, *, prefix: str = "") -> str:
        """Store bytes, return the storage key."""
        ...


class InMemoryObjectStore:
    """Dev/test store. Keeps bytes in a dict; nothing touches disk or network."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    async def put(self, data: bytes, content_type: str, *, prefix: str = "") -> str:
        key = _make_key(content_type, prefix)
        self._objects[key] = data
        return key

    def reset(self) -> None:
        self._objects.clear()


class LocalFileObjectStore:
    """Writes objects under a local directory. For local demos, not production."""

    def __init__(self, root: str) -> None:
        self._root = Path(root)

    async def put(self, data: bytes, content_type: str, *, prefix: str = "") -> str:
        key = _make_key(content_type, prefix)
        path = self._root / key
        await asyncio.to_thread(self._write, path, data)
        return key

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


class GCSObjectStore:
    """Google Cloud Storage. The google-cloud-storage client is blocking, so each
    upload runs in a thread to avoid stalling the event loop."""

    def __init__(self, bucket: str) -> None:
        from google.cloud import storage  # lazy: prod-only dependency

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)

    async def put(self, data: bytes, content_type: str, *, prefix: str = "") -> str:
        key = _make_key(content_type, prefix)
        await asyncio.to_thread(self._upload, key, data, content_type)
        return key

    def _upload(self, key: str, data: bytes, content_type: str) -> None:
        self._bucket.blob(key).upload_from_string(data, content_type=content_type)


_store: Optional[ObjectStore] = None


def get_object_store() -> ObjectStore:
    """Return the configured object store (cached singleton)."""
    global _store
    if _store is None:
        provider = settings.object_store_provider.strip().lower()
        if provider == "gcs":
            _store = GCSObjectStore(settings.gcs_bucket)
        elif provider == "local":
            _store = LocalFileObjectStore(settings.object_store_local_dir)
        else:
            _store = InMemoryObjectStore()
    return _store


def reset_object_store() -> None:
    """Test hook — drop the singleton so each test starts clean."""
    global _store
    if isinstance(_store, InMemoryObjectStore):
        _store.reset()
    _store = None
