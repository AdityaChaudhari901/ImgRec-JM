from datetime import date

import pytest

from app.services.ocr_parser import (
    calculate_days_since_expiry,
    normalize_ocr_dates,
)
from app.utils.date_utils import is_expired, parse_indian_date


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("20/05/2025", "2025-05-20"),   # DD/MM/YYYY
        ("20-05-2025", "2025-05-20"),   # DD-MM-YYYY
        ("2025-05-20", "2025-05-20"),   # already ISO
    ],
)
def test_parse_full_dates(raw, expected):
    assert parse_indian_date(raw) == expected


def test_month_year_only_snaps_to_last_day():
    # "JUN 2025" -> last day of June
    assert parse_indian_date("JUN 2025") == "2025-06-30"
    # February of a non-leap year -> 28
    assert parse_indian_date("FEB 2025") == "2025-02-28"


def test_unparseable_returns_none():
    assert parse_indian_date("not a date") is None
    assert parse_indian_date("") is None
    assert parse_indian_date(None) is None


def test_normalize_ocr_dates_coerces_indian_formats():
    ocr = {
        "manufacture_date": "JAN 2024",
        "expiry_date": "20/05/2025",
        "batch_no": "B2401K",
        "raw_text": "MFG JAN 2024 EXP 20/05/2025",
    }
    cleaned = normalize_ocr_dates(ocr)
    assert cleaned["manufacture_date"] == "2024-01-31"
    assert cleaned["expiry_date"] == "2025-05-20"
    # Non-date fields are preserved.
    assert cleaned["batch_no"] == "B2401K"
    assert cleaned["raw_text"].startswith("MFG")


def test_normalize_handles_null_strings():
    cleaned = normalize_ocr_dates({"manufacture_date": "null", "expiry_date": None})
    assert cleaned["manufacture_date"] is None
    assert cleaned["expiry_date"] is None


def test_days_since_expiry_positive_for_past_date():
    today = date(2026, 6, 17)
    assert calculate_days_since_expiry("2026-05-20", today=today) == 28


def test_days_since_expiry_none_for_future_date():
    today = date(2026, 6, 17)
    assert calculate_days_since_expiry("2027-01-01", today=today) is None


def test_days_since_expiry_none_for_bad_input():
    assert calculate_days_since_expiry(None) is None
    assert calculate_days_since_expiry("garbage") is None


def test_is_expired():
    today = date(2026, 6, 17)
    assert is_expired("2026-06-16", today=today) is True
    assert is_expired("2026-06-17", today=today) is False  # not strictly before
    assert is_expired("2026-06-18", today=today) is False
    assert is_expired(None, today=today) is False
