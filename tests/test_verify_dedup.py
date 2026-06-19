"""Phase 2 acceptance — reused-image fraud at the /verify-claim boundary.

The same (or lightly edited) photo submitted on a different account/order is flagged
as duplicate-fraud and rejected, with the matched prior claim ids recorded as the
justification — not an AI score.
"""

import base64
import io

from unittest.mock import patch

from PIL import Image
from httpx import ASGITransport, AsyncClient

from app.db import repository
from app.main import app

HEADERS = {"x-api-key": "test-secret", "Content-Type": "application/json"}

# A genuine-looking claim: high alignment + product match, no AI flag. On its own
# this auto-approves — so a reject can only come from the dedup hard signal.
GEMINI_GENUINE = {
    "recognition": {"scene": "leaking oil bottle", "objects": ["oil bottle"],
                    "extracted_text": "JioMart Oil"},
    "ai_generated": {"ai_probability": 0.04, "signals": []},
    "image_comment_alignment": {"score": 0.95, "aligned": True, "reason": "leak visible"},
    "product_match": {"detected_product": "oil", "matches": True, "score": 0.95, "reason": "match"},
    "other_flags": [],
    "summary": "Genuine claim.",
}


def _gradient_b64(seed: int, fmt: str = "PNG", quality: int = 90, size: int = 48) -> str:
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (x * seed + y * 3) % 256
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return "data:image/{};base64,{}".format(
        "jpeg" if fmt == "JPEG" else "png", base64.b64encode(buf.getvalue()).decode()
    )


def _claim(image_b64, order_id, user_id):
    return {
        "image_base64": image_b64,
        "user_comment": "It arrived damaged",
        "claimed_product": "JioMart Oil",
        "order_id": order_id,
        "user_id": user_id,
    }


async def _verify(client, payload):
    return await client.post("/api/v1/imgrecog/verify-claim", json=payload, headers=HEADERS)


async def test_same_image_across_two_users_is_rejected_as_duplicate_fraud():
    img = _gradient_b64(11)
    with patch("app.routers.verify.analyze_claim", return_value=GEMINI_GENUINE):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r1 = await _verify(c, _claim(img, "JM-100", "user_A"))
            r2 = await _verify(c, _claim(img, "JM-200", "user_B"))

    # First genuine claim auto-approves.
    assert r1.json()["recommended_action"] == "auto_approve"

    # Second (same photo, different user/order) is rejected as duplicate fraud.
    d2 = r2.json()
    assert d2["verdict"] == "likely_fraud"
    assert d2["recommended_action"] == "reject"
    # Justification is the concrete prior claim, not an AI probability.
    assert any("JM-100" in f for f in d2["checks"]["other_flags"])
    assert d2["checks"]["ai_generated"]["is_ai_generated"] is False

    # Audit record stores the matched claim ids as the reason.
    rows = list(repository.get_decision_repository()._by_key.values())
    rejected = [r for r in rows if r.order_id == "JM-200"][0]
    matched = rejected.computed["dedup"]["cross_claim_matches"]
    assert any(m["order_id"] == "JM-100" for m in matched)
    assert rejected.final_action == "reject"


async def test_lightly_edited_duplicate_is_still_caught():
    original = _gradient_b64(11, fmt="PNG")
    # Re-encode as lossy JPEG — bytes differ entirely, perceptual hash stays close.
    edited = _gradient_b64(11, fmt="JPEG", quality=30)
    with patch("app.routers.verify.analyze_claim", return_value=GEMINI_GENUINE):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await _verify(c, _claim(original, "JM-300", "user_C"))
            r2 = await _verify(c, _claim(edited, "JM-400", "user_D"))

    assert r2.json()["recommended_action"] == "reject"
    assert any("JM-300" in f for f in r2.json()["checks"]["other_flags"])


async def test_same_user_same_order_resubmission_is_not_fraud():
    # A near-duplicate from the SAME user+order (e.g. a re-shoot) is benign.
    original = _gradient_b64(11, fmt="PNG")
    edited = _gradient_b64(11, fmt="JPEG", quality=30)
    with patch("app.routers.verify.analyze_claim", return_value=GEMINI_GENUINE):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await _verify(c, _claim(original, "JM-500", "user_E"))
            r2 = await _verify(c, _claim(edited, "JM-500", "user_E"))

    assert r2.json()["recommended_action"] == "auto_approve"
