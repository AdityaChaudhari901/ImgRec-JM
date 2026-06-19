"""Phase 1 — durable audit + idempotency behaviour at the endpoint boundary.

Gemini is mocked at the router's `analyze_image` boundary (as elsewhere); the
audit store + object store use their in-memory implementations (reset per test by
the autouse fixture in conftest), so these tests need no DB, GCS, or API keys.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.db import repository
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

PAYLOAD = {
    "image_base64": "data:image/jpeg;base64,/9j/fake",
    "order_id": "JM-IDEM-1",
    "user_id": "u_idem",
    "scan_type": "auto",
}


async def _post(client, payload):
    return await client.post("/api/v1/imgrecog/scan", json=payload, headers=HEADERS)


@pytest.mark.asyncio
async def test_identical_scan_replays_and_calls_model_once():
    mock = AsyncMock(return_value=MOCK_EXPIRED)
    with patch("app.routers.scan.analyze_image", mock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r1 = await _post(client, PAYLOAD)
            r2 = await _post(client, PAYLOAD)

    assert r1.status_code == r2.status_code == 200
    # Verbatim replay: same decision id returned both times.
    assert r1.json()["request_id"] == r2.json()["request_id"]
    assert r1.json() == r2.json()
    # The model ran exactly once across the two identical requests.
    assert mock.call_count == 1


@pytest.mark.asyncio
async def test_each_call_persists_a_complete_audit_row():
    with patch("app.routers.scan.analyze_image", AsyncMock(return_value=MOCK_EXPIRED)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await _post(client, PAYLOAD)

    repo = repository.get_decision_repository()
    rows = list(repo._by_key.values())
    assert len(rows) == 1
    row = rows[0]
    assert row.endpoint == "scan"
    assert row.order_id == "JM-IDEM-1"
    assert row.final_action == "initiate_refund"
    assert row.routed_to == "auto"
    assert row.model_name and row.prompt_version
    assert row.image_ref  # points at object storage
    assert row.response_snapshot["request_id"] == row.request_id


@pytest.mark.asyncio
async def test_image_is_a_ref_not_a_blob_in_the_audit_row():
    with patch("app.routers.scan.analyze_image", AsyncMock(return_value=MOCK_EXPIRED)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await _post(client, PAYLOAD)

    row = list(repository.get_decision_repository()._by_key.values())[0]
    serialized = json.dumps(row.__dict__, default=str)
    # The base64 blob never lands in the DB record — only a storage key.
    assert "/9j/fake" not in serialized
    assert row.image_ref.endswith(".jpg")


@pytest.mark.asyncio
async def test_explicit_idempotency_key_dedups_even_with_a_different_image():
    p1 = {**PAYLOAD, "idempotency_key": "claim-ABC"}
    p2 = {**PAYLOAD, "idempotency_key": "claim-ABC",
          "image_base64": "data:image/png;base64,iVBORnotreal"}
    mock = AsyncMock(return_value=MOCK_EXPIRED)
    with patch("app.routers.scan.analyze_image", mock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r1 = await _post(client, p1)
            r2 = await _post(client, p2)

    assert mock.call_count == 1
    assert r1.json()["request_id"] == r2.json()["request_id"]


@pytest.mark.asyncio
async def test_audit_write_failure_downgrades_to_manual_review():
    repo = repository.get_decision_repository()
    with patch("app.routers.scan.analyze_image", AsyncMock(return_value=MOCK_EXPIRED)), \
         patch.object(repo, "insert", AsyncMock(side_effect=RuntimeError("db down"))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await _post(client, PAYLOAD)

    assert r.status_code == 200
    data = r.json()
    # A failed audit write must NEVER return an automated refund.
    assert data["action"]["type"] == "no_action"
    assert data["action"]["refund_eligible"] is False
    assert "manual review" in data["action"]["message"].lower()
