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


def test_different_categories_same_image_do_not_collide():
    # Same order + same image, two different disputes must NOT replay each other.
    base = {
        "images": ["data:image/jpeg;base64,AAAA"], "ticket": {"description": ""},
        "shipment": {"order_tracking_id": "JM-COL", "product_name": "Oil 1L", "product_type": "non_fnv",
                     "mrp": 100, "selling_price": 100, "invoice_amount": 100, "quantity": 1, "seller_type": "1P"},
    }

    async def mrp_obs(*a, **k):
        return {"ocr": {"printed_mrp_values": [90]}, "ai_generated": {"ai_probability": 0.0}}

    async def dmg_obs(*a, **k):
        return {"damage": {"detected": False}, "ai_generated": {"ai_probability": 0.0}}

    with patch("app.routers.dispute.analyze_dispute", side_effect=mrp_obs):
        r1 = client.post("/api/v1/imgrecog/dispute", json={**base, "dispute_category": "mrp_abuse"}, headers=HEADERS).json()
    with patch("app.routers.dispute.analyze_dispute", side_effect=dmg_obs):
        r2 = client.post("/api/v1/imgrecog/dispute", json={**base, "dispute_category": "damaged"}, headers=HEADERS).json()

    assert r1["category"] == "mrp_abuse" and r1["decision"] == "approve"
    assert r2["category"] == "damaged" and r2["decision"] == "reject"
    assert r1["request_id"] != r2["request_id"]


def test_rebuttal_not_replayed_from_original(monkeypatch):
    # A rebuttal on the same order+image+category must be processed (agent), not
    # replay the original decision.
    base = {
        "images": ["data:image/jpeg;base64,BBBB"], "dispute_category": "mrp_abuse", "ticket": {"description": "overcharged"},
        "shipment": {"order_tracking_id": "JM-RB", "product_name": "Oil 1L", "product_type": "non_fnv",
                     "mrp": 100, "selling_price": 100, "invoice_amount": 100, "quantity": 1, "seller_type": "1P"},
    }

    async def mrp_obs(*a, **k):
        return {"ocr": {"printed_mrp_values": [90]}, "ai_generated": {"ai_probability": 0.0}}

    with patch("app.routers.dispute.analyze_dispute", side_effect=mrp_obs):
        first = client.post("/api/v1/imgrecog/dispute", json=base, headers=HEADERS).json()
        rebut = client.post("/api/v1/imgrecog/dispute", json={**base, "is_rebuttal": True}, headers=HEADERS).json()

    assert first["decision"] == "approve"
    assert rebut["decision"] == "agent"
    assert "rebuttal" in rebut["agent_flags"]


def test_dispute_via_image_urls_no_shipment():
    from app.services.image_url_fetcher import FetchedImage

    async def fake_fetch(url, role="image"):
        return FetchedImage(source_url=url, fetched_url=url, mime_type="image/jpeg",
                            data=b"\xff\xd8\xff\xe0")

    async def fake_analyze(*a, **k):
        return {"damage": {"detected": True, "type": "leakage", "severity": "severe"},
                "ai_generated": {"ai_probability": 0.0}}

    body = {"image_urls": ["https://cdn.example.com/p.jpg"], "dispute_category": "damaged",
            "ticket": {"description": "bottle leaking"}}
    with patch("app.routers.dispute.download_image_url", side_effect=fake_fetch), \
         patch("app.routers.dispute.analyze_dispute", side_effect=fake_analyze):
        r = client.post("/api/v1/imgrecog/dispute", json=body, headers=HEADERS)
    assert r.status_code == 200
    d = r.json()
    assert d["category"] == "damaged" and d["decision"] == "approve"
    assert d["order_tracking_id"] == "no-order"


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
