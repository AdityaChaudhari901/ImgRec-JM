from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
HEADERS = {"x-api-key": "test-secret"}


def _body(**over):
    base = {
        "images": ["data:image/jpeg;base64,AAAA"],
        "dispute_category": "mrp_abuse",
        "ticket": {"title": "t", "description": "overcharged", "notes": "", "disposition_code": ""},
        "shipment": {"order_tracking_id": "JM-1", "product_name": "Oil 1L", "product_type": "non_fnv",
                     "mrp": 100.0, "selling_price": 100.0, "invoice_amount": 100.0,
                     "quantity": 1, "seller_type": "1P"},
    }
    base.update(over)
    return base


def test_dispute_requires_api_key():
    assert client.post("/api/v1/imgrecog/dispute", json=_body()).status_code == 401


def test_dispute_mrp_approve():
    obs = {"ocr": {"printed_mrp_values": [90.0]}, "ai_generated": {"ai_probability": 0.0}}

    async def fake_analyze(*a, **k):
        return obs

    with patch("app.routers.dispute.analyze_dispute", side_effect=fake_analyze):
        r = client.post("/api/v1/imgrecog/dispute", json=_body(), headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "mrp_abuse"
    assert data["decision"] == "approve"
    assert data["refund"]["amount"] == 10.0


def test_dispute_insufficient_data_to_agent():
    body = _body(dispute_category=None,
                 ticket={"title": "", "description": "", "notes": "", "disposition_code": ""})

    async def fake_analyze(*a, **k):
        return {"ocr": {}, "ai_generated": {"ai_probability": 0.0}}

    with patch("app.routers.dispute.analyze_dispute", side_effect=fake_analyze):
        r = client.post("/api/v1/imgrecog/dispute", json=body, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["decision"] == "agent"
    assert "insufficient_data" in data["agent_flags"]
