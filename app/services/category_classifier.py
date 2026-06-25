"""Resolve the dispute category via the requirement's §3 fallback chain.

Deterministic and model-free: explicit category -> keyword match on the
description -> keyword match on the notes -> disposition-code map -> None
(INSUFFICIENT_DATA). Keeping this in plain Python keeps it fast and testable.
"""

from typing import Optional, Tuple

from app.models.dispute_request import Ticket

# Order matters: earlier categories win when multiple keyword groups match.
_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("mrp_abuse", ("mrp", "overcharge", "over charge", "overcharged", "price", "charged more", "expensive")),
    ("expiry", ("expir", "expiry", "expired", "best before", "use by", "near expiry", "date over")),
    ("damaged", ("damage", "broken", "leak", "torn", "tear", "crush", "tamper", "seal", "spill", "dent")),
    ("smell", ("smell", "stink", "odor", "odour", "foul", "rotten")),
    ("wrong_product", ("wrong", "incorrect", "different item", "not what i ordered", "mismatch item")),
    ("quantity_mismatch", ("quantity", "missing item", "less item", "fewer", "short", "only got", "one less")),
    ("poor_quality", ("quality", "stale", "spoil", "bad", "wilt", "discolor", "fungus", "mold", "fresh")),
]

_DISPOSITION_MAP = {
    "WRONG_ITEM": "wrong_product",
    "QUALITY_ISSUE": "poor_quality",
    "DAMAGE": "damaged",
    "EXPIRY": "expiry",
    "PRICE_DISPUTE": "mrp_abuse",
}


def _match_keywords(text: str) -> Optional[str]:
    low = (text or "").lower()
    if not low.strip():
        return None
    for category, words in _KEYWORDS:
        if any(w in low for w in words):
            return category
    return None


def classify_category(
    dispute_category: Optional[str], ticket: Ticket
) -> Tuple[Optional[str], str]:
    if dispute_category:
        return dispute_category, "provided"
    by_desc = _match_keywords(ticket.description)
    if by_desc:
        return by_desc, "description"
    by_notes = _match_keywords(ticket.notes)
    if by_notes:
        return by_notes, "notes"
    mapped = _DISPOSITION_MAP.get((ticket.disposition_code or "").strip().upper())
    if mapped:
        return mapped, "disposition"
    return None, "none"
