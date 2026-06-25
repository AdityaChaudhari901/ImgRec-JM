from datetime import date

from app.services.ocr_parser import days_until_expiry, final_printed_mrp, shelf_left_pct


def test_final_printed_mrp_picks_lowest():
    assert final_printed_mrp([49.0, 31.0]) == 31.0


def test_final_printed_mrp_ignores_nonpositive_and_empty():
    assert final_printed_mrp([0, -5]) is None
    assert final_printed_mrp([]) is None


def test_days_until_expiry_future_and_past():
    today = date(2026, 6, 25)
    assert days_until_expiry("2026-07-10", today) == 15
    assert days_until_expiry("2026-06-20", today) == -5
    assert days_until_expiry(None, today) is None


def test_shelf_left_pct():
    today = date(2026, 6, 25)
    # mfg 2026-06-20, exp 2026-06-30 -> total 10d, 5 left -> 50%
    assert shelf_left_pct("2026-06-20", "2026-06-30", today) == 50.0
    assert shelf_left_pct(None, "2026-06-30", today) is None
