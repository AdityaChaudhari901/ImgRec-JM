import pytest
from pydantic import ValidationError

from app.models.dispute_request import DisputeRequest
from app.models.dispute_response import DisputeResponse, RefundResult


def _req(**over):
    base = dict(
        images=["data:image/jpeg;base64,AAAA"],
        ticket={"title": "t", "description": "wrong item", "notes": "", "disposition_code": ""},
        shipment={
            "order_tracking_id": "JM-1", "product_name": "Amul Milk",
            "product_type": "dairy", "mrp": 33.0, "selling_price": 31.0,
            "invoice_amount": 62.0, "quantity": 2, "seller_type": "1P",
        },
    )
    base.update(over)
    return DisputeRequest(**base)


def test_request_minimal_ok():
    r = _req()
    assert r.shipment.seller_type == "1P"
    assert r.is_rebuttal is False
    assert r.dispute_category is None


def test_request_rejects_empty_images():
    with pytest.raises(ValidationError):
        _req(images=[])


def test_request_rejects_bad_product_type():
    with pytest.raises(ValidationError):
        _req(shipment={**_req().shipment.model_dump(), "product_type": "gadget"})


def test_response_defaults():
    resp = DisputeResponse(
        request_id="dsp_1", order_tracking_id="JM-1", category="mrp_abuse",
        category_source="provided", decision="approve", route="auto",
        refund=RefundResult(eligible=True, amount=4.0, type="price_difference"),
        recommendation="ok", confidence=0.9, observations={}, model_used="gemini-2.5-flash",
    )
    assert resp.success is True
    assert resp.agent_flags == []
    assert resp.refund.assign_to_mpt is False
