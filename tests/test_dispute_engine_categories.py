from app.models.dispute_request import Shipment
from app.services.dispute_engine import decide

_NO = {"ai_probability": 0.0, "dedup_cross": False, "web_hard": False}


def _ship(pt="non_fnv", **o):
    base = dict(order_tracking_id="JM-1", product_name="Tata Salt 1kg", product_type=pt,
                mrp=28.0, selling_price=28.0, invoice_amount=28.0, quantity=3, seller_type="1P")
    base.update(o)
    return Shipment(**base)


def test_wrong_product_mismatch_approves():
    d = decide("wrong_product", "provided", {"product_match": {"matches": False}}, _ship(), False, _NO)
    assert d.decision == "approve"


def test_wrong_product_match_rejects():
    d = decide("wrong_product", "provided", {"product_match": {"matches": True}}, _ship(), False, _NO)
    assert d.decision == "reject"


def test_damaged_confirmed_approves():
    obs = {"damage": {"detected": True, "type": "leakage", "severity": "severe"}}
    d = decide("damaged", "provided", obs, _ship(), False, _NO)
    assert d.decision == "approve"


def test_damaged_intact_rejects():
    d = decide("damaged", "provided", {"damage": {"detected": False}}, _ship(), False, _NO)
    assert d.decision == "reject"


def test_poor_quality_internal_defect_to_agent():
    obs = {"quality": {"poor_quality": True, "supports_claim": True, "internal_defect": True}}
    d = decide("poor_quality", "provided", obs, _ship(), False, _NO)
    assert d.decision == "agent"
    assert "internal_defect" in d.agent_flags


def test_quantity_short_approves():
    obs = {"count": {"counted_units": 2, "confidence": 0.9}}  # ordered 3
    d = decide("quantity_mismatch", "provided", obs, _ship(), False, _NO)
    assert d.decision == "approve"


def test_quantity_low_confidence_agent():
    obs = {"count": {"counted_units": None, "confidence": 0.2}}
    d = decide("quantity_mismatch", "provided", obs, _ship(), False, _NO)
    assert d.decision == "agent"


def test_smell_with_spoilage_and_detail_approves():
    obs = {"spoilage": {"mold_or_visible_spoilage": True}, "_desc_len": 60}
    d = decide("smell", "provided", obs, _ship(), False, _NO)
    assert d.decision == "approve"
