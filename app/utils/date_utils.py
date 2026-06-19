"""Indian date-format parsing helpers.

Gemini already returns ISO dates in the happy path, but raw labels frequently
arrive in DD/MM/YYYY, MM/YYYY, MMM YYYY, DD-MM-YY, MON-YY, etc. These helpers
normalise any of those to ISO `YYYY-MM-DD`, applying the "last day of month"
rule when only a month/year is present.
"""

import calendar
import re
from datetime import date
from typing import Optional

import dateparser

# India reads dates day-first; force that so 03/04/2025 -> 3 April 2025.
_PARSER_SETTINGS = {
    "DATE_ORDER": "DMY",
    "PREFER_DAY_OF_MONTH": "last",
    "RETURN_AS_TIMEZONE_AWARE": False,
}

# Matches month/year only (no day), e.g. "JUN 2025", "06/2025", "JUN-25".
_MONTH_YEAR_ONLY = re.compile(
    r"^\s*(?:[A-Za-z]{3,9}|\d{1,2})[\s/\-]+\d{2,4}\s*$"
)
_HAS_DAY = re.compile(r"\b([0-3]?\d)[\s/\-]")


def parse_indian_date(raw: Optional[str]) -> Optional[str]:
    """Parse a single date token in any common Indian format to ISO `YYYY-MM-DD`.

    Returns None if the token cannot be understood.
    """
    if not raw:
        return None

    token = raw.strip()
    if not token:
        return None

    # Fast path: already ISO (YYYY-MM-DD). dateparser with DMY ordering would
    # otherwise misread the year as a day and return None.
    try:
        return date.fromisoformat(token).isoformat()
    except ValueError:
        pass

    parsed = dateparser.parse(token, settings=_PARSER_SETTINGS)
    if parsed is None:
        return None

    result = parsed.date()

    # Month/year only -> snap to the last day of that month.
    if _MONTH_YEAR_ONLY.match(token) and not _HAS_DAY.search(token):
        last_day = calendar.monthrange(result.year, result.month)[1]
        result = result.replace(day=last_day)

    return result.isoformat()


def is_expired(expiry_iso: Optional[str], today: Optional[date] = None) -> bool:
    """True if the ISO expiry date is strictly before today."""
    if not expiry_iso:
        return False
    try:
        expiry = date.fromisoformat(expiry_iso)
    except ValueError:
        return False
    return expiry < (today or date.today())
