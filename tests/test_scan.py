import pytest
from httpx import ASGITransport, AsyncClient

from unittest.mock import patch

from app.main import app

HEADERS = {"x-api-key": "test-secret", "Content-Type": "application/json"}

MOCK_EXPIRED = {
    "status": "expired",
    "confidence": 0.96,
    "ocr": {
        "manufacture_date": "2024-01-15",
        "expiry_date": "2023-05-20",
        "batch_no": "B2401K",
        "raw_text": "MFG JAN 2024 EXP MAY 2023",
    },
    "damage": {"detected": False, "type": None, "severity": None, "description": None},
    "ai_generated": {"ai_probability": 0.03, "signals": []},
    "action": {
        "type": "initiate_refund",
        "message": "Expired.",
        "refund_eligible": True,
        "priority": "high",
    },
}

MOCK_DAMAGED = {
    "status": "damaged",
    "confidence": 0.91,
    "ocr": {
        "manufacture_date": "2025-03-01",
        "expiry_date": "2026-03-01",
        "batch_no": None,
        "raw_text": "MFG MAR 2025",
    },
    "damage": {
        "detected": True,
        "type": "crushed_packaging",
        "severity": "moderate",
        "description": "Box crushed",
    },
    "ai_generated": {"ai_probability": 0.04, "signals": []},
    "action": {
        "type": "initiate_exchange",
        "message": "Damaged.",
        "refund_eligible": False,
        "priority": "medium",
    },
}

MOCK_AI_DAMAGED = {
    "status": "damaged",
    "confidence": 0.90,
    "ocr": {
        "manufacture_date": None,
        "expiry_date": None,
        "batch_no": None,
        "raw_text": "AI milk label",
    },
    "damage": {
        "detected": True,
        "type": "leakage",
        "severity": "severe",
        "description": "Milk appears to be leaking from the package",
    },
    "ai_generated": {
        "ai_probability": 0.92,
        "signals": ["warped label text", "synthetic liquid texture"],
    },
    "action": {
        "type": "initiate_refund",
        "message": "We apologize for the damaged product. A refund has been initiated.",
        "refund_eligible": True,
        "priority": "high",
    },
}

MOCK_UNCLEAR_DAMAGE_REFUND_MESSAGE = {
    "status": "unclear",
    "confidence": 0.70,
    "ocr": {
        "manufacture_date": None,
        "expiry_date": None,
        "batch_no": None,
        "raw_text": "warped Amul Taaza milk OCR text",
    },
    "damage": {
        "detected": True,
        "type": "leakage",
        "severity": "severe",
        "description": "The milk pouch is torn and leaking.",
    },
    "action": {
        "type": "no_action",
        "message": "We are sorry your product arrived damaged. We will initiate a full refund for this item.",
        "refund_eligible": False,
        "priority": "low",
    },
}

MOCK_DAMAGED_REFUND_WITHOUT_AI_ASSESSMENT = {
    "status": "damaged",
    "confidence": 0.90,
    "ocr": {
        "manufacture_date": None,
        "expiry_date": None,
        "batch_no": None,
        "raw_text": "warped Amul Taaza milk OCR text",
    },
    "damage": {
        "detected": True,
        "type": "leakage",
        "severity": "severe",
        "description": "The milk pouch is torn and leaking.",
    },
    "action": {
        "type": "initiate_refund",
        "message": "We are sorry your item arrived damaged. A refund has been initiated for this product.",
        "refund_eligible": True,
        "priority": "high",
    },
}

VALID_PAYLOAD = {
    "image_base64": "data:image/jpeg;base64,/9j/fake",
    "order_id": "JM-001",
    "user_id": "u_123",
    "scan_type": "auto",
}


@pytest.mark.asyncio
async def test_expired_product_returns_refund():
    with patch("app.routers.scan.analyze_image", return_value=MOCK_EXPIRED):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/v1/imgrecog/scan", json=VALID_PAYLOAD, headers=HEADERS
            )
    assert r.status_code == 200
    assert r.json()["status"] == "expired"
    assert r.json()["action"]["type"] == "initiate_refund"


@pytest.mark.asyncio
async def test_damaged_product_returns_exchange():
    with patch("app.routers.scan.analyze_image", return_value=MOCK_DAMAGED):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/v1/imgrecog/scan", json=VALID_PAYLOAD, headers=HEADERS
            )
    assert r.status_code == 200
    assert r.json()["damage"]["detected"] is True
    assert r.json()["action"]["type"] == "initiate_exchange"


@pytest.mark.asyncio
async def test_ai_generated_damaged_product_does_not_trigger_refund():
    with patch("app.routers.scan.analyze_image", return_value=MOCK_AI_DAMAGED):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/v1/imgrecog/scan", json=VALID_PAYLOAD, headers=HEADERS
            )

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "unclear"
    assert data["damage"]["detected"] is False
    assert data["ai_generated"]["is_ai_generated"] is True
    assert data["ai_generated"]["ai_probability"] == 0.92
    assert data["action"]["type"] == "no_action"
    assert data["action"]["refund_eligible"] is False
    assert data["action"]["priority"] == "high"
    assert "manual review" in data["action"]["message"]


@pytest.mark.asyncio
async def test_unclear_damage_refund_message_routes_to_manual_review():
    with patch(
        "app.routers.scan.analyze_image",
        return_value=MOCK_UNCLEAR_DAMAGE_REFUND_MESSAGE,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/v1/imgrecog/scan", json=VALID_PAYLOAD, headers=HEADERS
            )

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "unclear"
    assert data["damage"]["detected"] is False
    assert data["damage"]["type"] is None
    assert data["action"]["type"] == "no_action"
    assert data["action"]["refund_eligible"] is False
    assert data["action"]["priority"] == "high"
    assert "manual review" in data["action"]["message"]
    assert "full refund for this item" not in data["action"]["message"]


@pytest.mark.asyncio
async def test_damaged_refund_without_ai_assessment_routes_to_manual_review():
    with patch(
        "app.routers.scan.analyze_image",
        return_value=MOCK_DAMAGED_REFUND_WITHOUT_AI_ASSESSMENT,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/v1/imgrecog/scan", json=VALID_PAYLOAD, headers=HEADERS
            )

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "unclear"
    assert data["damage"]["detected"] is False
    assert data["action"]["type"] == "no_action"
    assert data["action"]["refund_eligible"] is False
    assert data["action"]["priority"] == "high"
    assert "authenticity was not assessed" in data["action"]["message"]
    assert "refund has been initiated" not in data["action"]["message"]


@pytest.mark.asyncio
async def test_missing_image_returns_422():
    payload = {"order_id": "JM-001", "user_id": "u_123"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/api/v1/imgrecog/scan", json=payload, headers=HEADERS
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_wrong_api_key_returns_401():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/api/v1/imgrecog/scan",
            json=VALID_PAYLOAD,
            headers={"x-api-key": "wrong-key"},
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_gemini_timeout_returns_504():
    with patch("app.routers.scan.analyze_image", side_effect=TimeoutError("timeout")):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post(
                "/api/v1/imgrecog/scan", json=VALID_PAYLOAD, headers=HEADERS
            )
    assert r.status_code == 504
