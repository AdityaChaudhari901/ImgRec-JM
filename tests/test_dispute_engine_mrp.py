from app.models.dispute_request import Shipment
from app.services.dispute_engine import decide


def _ship(**o):
    base = dict(order_tracking_id="JM-1", product_name="Oil 1L", product_type="non_fnv",
                mrp=100.0, selling_price=100.0, invoice_amount=200.0, quantity=2, seller_type="1P")
    base.update(o)
    return Shipment(**base)


_NO_SIGNALS = {"ai_probability": 0.0, "dedup_cross": False, "web_hard": False}


def test_mrp_overcharge_1p_refunds_difference():
    obs = {"ocr": {"printed_mrp_values": [90.0]}}  # printed 90 < invoice mrp 100 -> overcharge
    d = decide("mrp_abuse", "provided", obs, _ship(), False, _NO_SIGNALS)
    assert d.decision == "approve"
    assert d.refund["type"] == "price_difference"
    assert d.refund["amount"] == 20.0  # (100 charged - 90 printed) * 2
    assert d.refund["assign_to_mpt"] is False


def test_mrp_no_overcharge_rejects():
    obs = {"ocr": {"printed_mrp_values": [100.0]}}  # printed == invoice mrp -> reject
    d = decide("mrp_abuse", "provided", obs, _ship(), False, _NO_SIGNALS)
    assert d.decision == "reject"
    assert d.refund["eligible"] is False


def test_mrp_3p_full_refund_and_mpt():
    obs = {"ocr": {"printed_mrp_values": [90.0]}}
    d = decide("mrp_abuse", "provided", obs, _ship(seller_type="3P"), False, _NO_SIGNALS)
    assert d.decision == "approve"  # 100*2 = 200 < 500 ceiling and mrp_abuse is autonomous
    assert d.refund["type"] == "full_selling_price"
    assert d.refund["amount"] == 200.0
    assert d.refund["assign_to_mpt"] is True
    assert d.refund["seller_debit"] is True


def test_mrp_unreadable_routes_agent():
    obs = {"ocr": {"printed_mrp_values": []}}
    d = decide("mrp_abuse", "provided", obs, _ship(), False, _NO_SIGNALS)
    assert d.decision == "agent"
    assert "missing_shipment_data" in d.agent_flags or "low_confidence" in d.agent_flags


def test_mrp_discount_below_printed_is_not_overcharge():
    # Regression for the old printed-vs-recorded-MRP bug: charged BELOW the printed
    # MRP (a discount) must reject, even though the recorded MRP differs.
    obs = {"ocr": {"printed_mrp_values": [90.0]}}
    # invoice 170 / qty 2 = 85 charged < printed 90 -> no overcharge
    d = decide("mrp_abuse", "provided", obs, _ship(invoice_amount=170.0, quantity=2), False, _NO_SIGNALS)
    assert d.decision == "reject"
    assert d.refund["eligible"] is False


def test_mrp_charged_above_printed_uses_invoice_amount():
    obs = {"ocr": {"printed_mrp_values": [90.0]}}
    # invoice 240 / qty 2 = 120 charged > printed 90 -> overcharge of 30/unit
    d = decide("mrp_abuse", "provided", obs, _ship(invoice_amount=240.0, quantity=2), False, _NO_SIGNALS)
    assert d.decision == "approve"
    assert d.refund["amount"] == 60.0  # (120 - 90) * 2


def test_mrp_without_shipment_routes_agent():
    obs = {"ocr": {"printed_mrp_values": [90.0]}}
    d = decide("mrp_abuse", "provided", obs, None, False, _NO_SIGNALS)
    assert d.decision == "agent"
    assert "missing_shipment_data" in d.agent_flags
