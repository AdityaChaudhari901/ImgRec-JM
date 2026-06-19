import pytest
from httpx import ASGITransport, AsyncClient

from unittest.mock import patch

from app.main import app

HEADERS = {"x-api-key": "test-secret", "Content-Type": "application/json"}

GEMINI_AUTHENTIC = {
    "recognition": {"scene": "A leaking oil bottle on a table", "objects": ["oil bottle"],
                    "extracted_text": "JioMart Sunflower Oil 1L MFG 2026"},
    "ai_generated": {"ai_probability": 0.05, "signals": []},
    "image_comment_alignment": {"score": 0.93, "aligned": True, "reason": "Leak visible"},
    "product_match": {"detected_product": "JioMart Sunflower Oil 1L", "matches": True,
                      "score": 0.9, "reason": "Same bottle"},
    "other_flags": [],
    "summary": "Genuine leaking oil bottle matching the claim.",
}

GEMINI_AI_FAKE = {
    "ai_generated": {"ai_probability": 0.92, "signals": ["warped label text"]},
    "image_comment_alignment": {"score": 0.8, "aligned": True, "reason": "looks aligned"},
    "product_match": {"detected_product": "oil bottle", "matches": True, "score": 0.8, "reason": "ok"},
    "other_flags": [],
    "summary": "Image appears AI-generated.",
}

PAYLOAD = {
    "image_base64": "data:image/jpeg;base64,/9j/fake",
    "user_comment": "The oil bottle was leaking when it arrived",
    "claimed_product": "JioMart Sunflower Oil 1L",
    "order_id": "JM-77",
    "user_id": "u_9",
}


@pytest.mark.asyncio
async def test_authentic_claim_auto_approves():
    with patch("app.routers.verify.analyze_claim", return_value=GEMINI_AUTHENTIC):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/v1/imgrecog/verify-claim", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["verdict"] == "authentic"
    assert data["recommended_action"] == "auto_approve"
    assert data["authenticity_score"] >= 0.75
    assert data["checks"]["ai_generated"]["is_ai_generated"] is False
    # Image + text recognition (OCR) is returned.
    assert data["recognition"]["extracted_text"] == "JioMart Sunflower Oil 1L MFG 2026"
    assert "oil bottle" in data["recognition"]["objects"]


@pytest.mark.asyncio
async def test_ai_generated_image_routes_to_manual_review():
    # Internal detector is advisory -> manual_review, never auto-reject.
    with patch("app.routers.verify.analyze_claim", return_value=GEMINI_AI_FAKE):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/v1/imgrecog/verify-claim", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["checks"]["ai_generated"]["is_ai_generated"] is True
    assert data["recommended_action"] == "manual_review"


@pytest.mark.asyncio
async def test_missing_comment_returns_422():
    bad = {k: v for k, v in PAYLOAD.items() if k != "user_comment"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/imgrecog/verify-claim", json=bad, headers=HEADERS)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_wrong_api_key_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/imgrecog/verify-claim", json=PAYLOAD,
                         headers={"x-api-key": "nope"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_timeout_returns_504():
    with patch("app.routers.verify.analyze_claim", side_effect=TimeoutError("t")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/v1/imgrecog/verify-claim", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 504
