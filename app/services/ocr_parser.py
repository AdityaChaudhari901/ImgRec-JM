"""Date extraction + expiry calculation.

Gemini does the visual OCR; this module is the deterministic post-processor that
(a) normalises whatever date strings come back into ISO format and (b) computes
how long ago the product expired. Keeping this in plain Python (not the prompt)
means the expiry maths is testable and never hallucinated.
"""

from datetime import date
from typing import Optional

from app.utils.date_utils import parse_indian_date


def normalize_ocr_dates(ocr: dict) -> dict:
    """Return a copy of the OCR dict with manufacture/expiry coerced to ISO.

    Values already in ISO pass through unchanged; anything else is run through
    the Indian-format parser. Unparseable values become None.
    """
    cleaned = dict(ocr or {})
    for key in ("manufacture_date", "expiry_date"):
        cleaned[key] = _to_iso(cleaned.get(key))
    return cleaned


def _to_iso(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"null", "none", "n/a", "unknown"}:
        return None
    # Already ISO? keep it.
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return parse_indian_date(value)


def calculate_days_since_expiry(
    expiry_iso: Optional[str], today: Optional[date] = None
) -> Optional[int]:
    """Whole days since expiry. None if not expired or date is unusable."""
    if not expiry_iso:
        return None
    try:
        expiry = date.fromisoformat(expiry_iso)
    except ValueError:
        return None
    delta = ((today or date.today()) - expiry).days
    return delta if delta > 0 else None


def final_printed_mrp(values: object) -> Optional[float]:
    """Pick the final (post-strikethrough) MRP from OCR'd candidate values.

    A reduced-price pack prints both the old (struck-through, higher) and the
    new (lower) MRP. The lowest positive value is the one the customer pays
    against, so it is the right basis for an overcharge comparison.
    """
    if not isinstance(values, (list, tuple)):
        return None
    nums: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            nums.append(f)
    return min(nums) if nums else None


def days_until_expiry(expiry_iso: Optional[str], today: Optional[date] = None) -> Optional[int]:
    """Whole days from today until expiry. Negative if already expired, None if unusable."""
    if not expiry_iso:
        return None
    try:
        expiry = date.fromisoformat(expiry_iso)
    except ValueError:
        return None
    return (expiry - (today or date.today())).days


def shelf_left_pct(
    mfg_iso: Optional[str], exp_iso: Optional[str], today: Optional[date] = None
) -> Optional[float]:
    """Remaining shelf life as a percent of total shelf life. None if unusable."""
    if not mfg_iso or not exp_iso:
        return None
    try:
        mfg = date.fromisoformat(mfg_iso)
        exp = date.fromisoformat(exp_iso)
    except ValueError:
        return None
    total = (exp - mfg).days
    if total <= 0:
        return None
    remaining = (exp - (today or date.today())).days
    return round(remaining / total * 100, 2)
