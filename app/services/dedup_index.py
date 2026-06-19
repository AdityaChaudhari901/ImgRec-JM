"""Pluggable perceptual-hash dedup index.

Finds near-duplicate images (Hamming distance <= threshold on the 64-bit dHash)
seen on prior claims. Backends, selected by REDIS_URL:

  - InMemoryDedupIndex : dev/tests, no Redis needed.
  - RedisDedupIndex    : production (shared across serverless instances).

Both use the same **band (LSH) index**: the 16-hex-char hash is split into its 16
nibbles; two hashes within Hamming distance K must share at least one (position,
nibble) band when K < 16 (pigeonhole), so we fetch only candidates sharing a band,
then verify the exact Hamming distance. This keeps lookups sublinear without a full
scan. The lookback window is enforced by entry TTL (Redis) / timestamp filter (memory).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Protocol

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DuplicateMatch:
    order_id: str
    user_id: str
    request_id: str
    hamming: int
    created_at: float  # epoch seconds


def hamming_hex(a: str, b: str) -> int:
    """Hamming distance between two equal-length hex hashes (bit differences)."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _bands(phash: str) -> List[str]:
    """16 (position, nibble) bands — one per hex char."""
    return [f"{i}:{ch}" for i, ch in enumerate(phash)]


class DedupIndex(Protocol):
    async def add(self, phash: str, order_id: str, user_id: str, request_id: str) -> None:
        ...

    async def query(self, phash: str, threshold: int) -> List[DuplicateMatch]:
        ...


@dataclass
class _Entry:
    phash: str
    order_id: str
    user_id: str
    request_id: str
    created_at: float


class InMemoryDedupIndex:
    """Process-local band index for dev/tests."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._bands: dict[str, set[str]] = defaultdict(set)

    async def add(self, phash: str, order_id: str, user_id: str, request_id: str) -> None:
        entry = _Entry(phash, order_id, user_id, request_id, time.time())
        self._entries[request_id] = entry
        for band in _bands(phash):
            self._bands[band].add(request_id)

    async def query(self, phash: str, threshold: int) -> List[DuplicateMatch]:
        candidates: set[str] = set()
        for band in _bands(phash):
            candidates |= self._bands.get(band, set())

        cutoff = time.time() - settings.dedup_window_days * 86400
        out: List[DuplicateMatch] = []
        for rid in candidates:
            entry = self._entries.get(rid)
            if entry is None or entry.created_at < cutoff:
                continue
            dist = hamming_hex(phash, entry.phash)
            if dist <= threshold:
                out.append(
                    DuplicateMatch(
                        entry.order_id, entry.user_id, entry.request_id, dist, entry.created_at
                    )
                )
        return out

    def reset(self) -> None:
        self._entries.clear()
        self._bands.clear()


class RedisDedupIndex:
    """Shared band index in Redis. Entries carry a window-length TTL; band-set
    members that point at expired entries are skipped on read."""

    def __init__(self, url: str) -> None:
        import redis.asyncio as redis  # lazy: prod-only dependency

        self._r = redis.from_url(url, decode_responses=True)
        self._ttl = settings.dedup_window_days * 86400

    async def add(self, phash: str, order_id: str, user_id: str, request_id: str) -> None:
        entry_key = f"dedup:entry:{request_id}"
        pipe = self._r.pipeline()
        pipe.hset(
            entry_key,
            mapping={
                "phash": phash,
                "order_id": order_id,
                "user_id": user_id,
                "request_id": request_id,
                "created_at": time.time(),
            },
        )
        pipe.expire(entry_key, self._ttl)
        for band in _bands(phash):
            band_key = f"dedup:band:{band}"
            pipe.sadd(band_key, request_id)
            pipe.expire(band_key, self._ttl)
        await pipe.execute()

    async def query(self, phash: str, threshold: int) -> List[DuplicateMatch]:
        candidates: set[str] = set()
        for band in _bands(phash):
            candidates |= set(await self._r.smembers(f"dedup:band:{band}"))

        out: List[DuplicateMatch] = []
        for rid in candidates:
            entry = await self._r.hgetall(f"dedup:entry:{rid}")
            if not entry:  # expired -> outside the window
                continue
            dist = hamming_hex(phash, entry["phash"])
            if dist <= threshold:
                out.append(
                    DuplicateMatch(
                        entry["order_id"],
                        entry["user_id"],
                        entry["request_id"],
                        dist,
                        float(entry["created_at"]),
                    )
                )
        return out


_index: Optional[DedupIndex] = None


def get_dedup_index() -> DedupIndex:
    global _index
    if _index is None:
        if settings.redis_url:
            _index = RedisDedupIndex(settings.redis_url)
        else:
            logger.warning(
                "dedup_index_in_memory",
                detail="REDIS_URL unset — using non-shared in-memory dedup index",
            )
            _index = InMemoryDedupIndex()
    return _index


def reset_dedup_index() -> None:
    """Test hook — drop the singleton so each test starts clean."""
    global _index
    if isinstance(_index, InMemoryDedupIndex):
        _index.reset()
    _index = None
