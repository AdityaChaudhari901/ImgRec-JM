from datetime import date

from app.models.dispute_request import Shipment
from app.services.dispute_engine import decide

_NO = {"ai_probability": 0.0, "dedup_cross": False, "web_hard": False}


def _ship(pt="non_fnv", **o):
    base = dict(order_tracking_id="JM-1", product_name="Biscuits", product_type=pt,
                mrp=50.0, selling_price=50.0, invoice_amount=50.0, quantity=1, seller_type="1P")
    base.update(o)
    return Shipment(**base)


def test_non_fnv_near_expiry_approves(monkeypatch):
    from app.services import dispute_engine
    obs = {"ocr": {"expiry_date": "2026-07-01"}}  # ~6 days from 2026-06-25
    monkeypatch.setattr(dispute_engine, "_today", lambda: date(2026, 6, 25))
    d = decide("expiry", "provided", obs, _ship(), False, _NO)
    assert d.decision == "approve"


def test_non_fnv_far_expiry_rejects(monkeypatch):
    from app.services import dispute_engine
    obs = {"ocr": {"expiry_date": "2026-12-31"}}
    monkeypatch.setattr(dispute_engine, "_today", lambda: date(2026, 6, 25))
    d = decide("expiry", "provided", obs, _ship(), False, _NO)
    assert d.decision == "reject"


def test_dairy_low_shelf_approves(monkeypatch):
    from app.services import dispute_engine
    obs = {"ocr": {"manufacture_date": "2026-06-20", "expiry_date": "2026-06-30"}}  # 10d total
    monkeypatch.setattr(dispute_engine, "_today", lambda: date(2026, 6, 28))
    d = decide("expiry", "provided", obs, _ship(pt="dairy"), False, _NO)  # 2/10 = 20% < 30
    assert d.decision == "approve"


def test_expiry_unreadable_agent():
    d = decide("expiry", "provided", {"ocr": {}}, _ship(), False, _NO)
    assert d.decision == "agent"
