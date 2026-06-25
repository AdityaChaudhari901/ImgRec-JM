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
