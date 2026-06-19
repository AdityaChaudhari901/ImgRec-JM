"""Reused-image dedup: the highest-ROI fraud control.

Looks up whether a claim's photo (by perceptual hash) was already used on another
claim, and classifies the result relative to the *current* claim:

  - cross_claim_matches : the same image on a DIFFERENT order or user. This is a
    HARD, auditable fraud signal — it may drive an automated reject (principle #2:
    "your photo matched another claim" is defensible; "an AI thought it looked fake"
    is not).
  - same_claim_matches  : the same image re-submitted on the SAME order+user — a
    benign duplicate/resubmission, not fraud.

The index itself lives in `dedup_index` (in-memory dev / Redis prod). This module
adds the classification + resilient wrappers so an index outage degrades to "no
duplicate signal" (we may miss a catch, but never crash or wrongly auto-reject).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.services.dedup_index import (
    DuplicateMatch,
    get_dedup_index,
    hamming_hex,  # re-exported for tests/callers
)
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["DuplicateMatch", "DedupResult", "find_duplicates", "register_image", "hamming_hex"]


@dataclass
class DedupResult:
    cross_claim_matches: List[DuplicateMatch] = field(default_factory=list)
    same_claim_matches: List[DuplicateMatch] = field(default_factory=list)

    @property
    def is_cross_claim_duplicate(self) -> bool:
        return bool(self.cross_claim_matches)

    @classmethod
    def classify(
        cls, matches: List[DuplicateMatch], *, order_id: str, user_id: str
    ) -> "DedupResult":
        cross, same = [], []
        for m in matches:
            if m.order_id == order_id and m.user_id == user_id:
                same.append(m)
            else:
                cross.append(m)
        return cls(cross_claim_matches=cross, same_claim_matches=same)

    def matched_order_ids(self) -> List[str]:
        """Distinct prior order ids behind a cross-claim flag — the audit justification."""
        return sorted({m.order_id for m in self.cross_claim_matches})

    def to_audit(self) -> dict:
        def _dump(m: DuplicateMatch) -> dict:
            return {
                "order_id": m.order_id,
                "user_id": m.user_id,
                "request_id": m.request_id,
                "hamming": m.hamming,
            }

        return {
            "cross_claim_matches": [_dump(m) for m in self.cross_claim_matches],
            "same_claim_matches": [_dump(m) for m in self.same_claim_matches],
        }


async def find_duplicates(
    phash: Optional[str], order_id: str, user_id: str
) -> DedupResult:
    """Look up near-duplicates and classify them. Resilient: an index error yields
    an empty result (degrade to no-duplicate-signal) rather than failing the request."""
    if not phash:
        return DedupResult()
    try:
        matches = await get_dedup_index().query(phash, settings.dedup_hamming_threshold)
    except Exception as exc:  # noqa: BLE001 - dedup outage must not 500 the claim
        logger.error("dedup_query_failed", error=str(exc), order_id=order_id)
        return DedupResult()
    return DedupResult.classify(matches, order_id=order_id, user_id=user_id)


async def register_image(
    phash: Optional[str], order_id: str, user_id: str, request_id: str
) -> None:
    """Record this claim's image so future claims can match it. Best-effort."""
    if not phash:
        return
    try:
        await get_dedup_index().add(phash, order_id, user_id, request_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("dedup_register_failed", error=str(exc), order_id=order_id)
