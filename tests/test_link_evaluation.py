from datetime import date
from io import BytesIO
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.models.verify_response import AIGeneratedCheck
from app.services.image_url_fetcher import FetchedImage
from app.services.link_evaluation_service import build_link_evaluation_response
from app.services.web_provenance import WebProvenanceResult

HEADERS = {"x-api-key": "test-secret", "Content-Type": "application/json"}


def _png_bytes(color: str) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _fetched(url: str, color: str = "white") -> FetchedImage:
    return FetchedImage(
        source_url=url,
        fetched_url=url,
        mime_type="image/png",
        data=_png_bytes(color),
    )


OBSERVATIONS = {
    "product_status": {
        "status": "damaged",
        "score": 0.91,
        "verdict": "Product appears damaged.",
        "detail": "Bottle cap area appears broken.",
        "evidence": ["broken cap area"],
    },
    "authenticity": {
        "mobile_capture_score": 0.9,
        "ai_generated": {"ai_probability": 0.04, "signals": ["natural shadows"]},
        "verdict": "Image appears to be a real customer photo.",
        "detail": "Natural background and perspective.",
    },
    "product_match": {
        "score": 0.96,
        "matches": True,
        "detected_product": "JioMart sunflower oil bottle",
        "reference_product": "JioMart sunflower oil bottle",
        "verdict": "The same product and packaging are visible.",
        "detail": "Brand, bottle shape, and label colors match.",
        "match_evidence": ["same logo", "same bottle shape"],
        "mismatch_evidence": [],
    },
    "query_match": {
        "score": 0.9,
        "matches": True,
        "query_type": "damaged",
        "verdict": "Visible damage supports the query.",
        "detail": "Bottle cap area appears broken.",
        "evidence": ["broken cap area"],
    },
}


@pytest.mark.asyncio
async def test_evaluate_links_returns_auto_status_and_three_scores():
    user = _fetched("https://cdn.example.com/user.png", "red")
    product = _fetched("https://cdn.example.com/product.png", "blue")

    async def fake_download(url, role):
        return user if role == "user image" else product

    with (
        patch("app.routers.link_evaluation.download_image_url", new=fake_download),
        patch("app.routers.link_evaluation.analyze_linked_images", return_value=OBSERVATIONS),
        patch(
            "app.routers.link_evaluation.detect_ai_generated",
            return_value=AIGeneratedCheck(
                is_ai_generated=False,
                ai_probability=0.02,
                source="internal",
                signals=[],
            ),
        ),
        patch(
            "app.routers.link_evaluation.detect_web_provenance",
            return_value=WebProvenanceResult(checked=True),
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/api/v1/imgrecog/evaluate-links",
                headers=HEADERS,
                json={
                    "user_image_url": "https://cdn.example.com/user.png",
                    "product_image_url": "https://cdn.example.com/product.png",
                    "query": "damaged product",
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "decision",
        "product_status",
        "authenticity",
        "product_match",
        "query_match",
    }
    assert body["decision"]["decision"] == "accept_claim"
    assert body["decision"]["reason_codes"] == ["damaged_claim_supported"]
    assert body["product_status"]["status"] == "damaged"
    assert body["product_status"]["score"] == 91
    assert body["authenticity"]["score"] >= 90
    assert body["product_match"]["score"] == 96
    assert body["query_match"]["score"] == 90
    assert "same product" in body["product_match"]["verdict"].lower()


@pytest.mark.asyncio
async def test_evaluate_links_rejects_private_url_before_network():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post(
            "/api/v1/imgrecog/evaluate-links",
            headers=HEADERS,
            json={
                "user_image_url": "http://127.0.0.1/user.png",
                "product_image_url": "http://127.0.0.1/product.png",
                "query": "expired product",
            },
        )

    assert response.status_code == 422
    assert "non-public" in response.json()["detail"]


def test_web_download_match_zeroes_authenticity_score():
    result = build_link_evaluation_response(
        OBSERVATIONS,
        AIGeneratedCheck(
            is_ai_generated=False,
            ai_probability=0.02,
            source="internal",
            signals=[],
        ),
        WebProvenanceResult(checked=True, full_match_count=2, distinct_domains=2),
    )

    assert result.authenticity.score == 0
    assert "web" in result.authenticity.verdict.lower()
    assert result.decision.decision == "review"
    assert result.decision.reason_codes == ["authenticity_below_threshold"]


def test_product_match_score_mapping_is_conservative_without_model_verdict():
    obs = {
        "product_status": {"status": "valid", "score": 0.7},
        "authenticity": {"mobile_capture_score": 0.8, "ai_generated": {"ai_probability": 0.1}},
        "product_match": {"score": 0.62},
        "query_match": {"score": 0.91},
    }

    result = build_link_evaluation_response(
        obs,
        AIGeneratedCheck(
            is_ai_generated=False,
            ai_probability=0.1,
            source="internal",
            signals=[],
        ),
        WebProvenanceResult(checked=False),
    )

    assert result.product_match.score == 62
    assert "weak" in result.product_match.verdict.lower()
    assert result.decision.decision == "review"
    assert result.decision.reason_codes == ["product_match_below_threshold"]


def test_invalid_product_status_falls_back_to_unclear():
    obs = {
        "product_status": {"status": "fresh-ish", "score": 0.8},
        "authenticity": {"mobile_capture_score": 0.8, "ai_generated": {"ai_probability": 0.1}},
        "product_match": {"score": 0.9},
        "query_match": {"score": 0.9},
    }

    result = build_link_evaluation_response(
        obs,
        AIGeneratedCheck(
            is_ai_generated=False,
            ai_probability=0.1,
            source="internal",
            signals=[],
        ),
        WebProvenanceResult(checked=False),
    )

    assert result.product_status.status == "unclear"
    assert result.product_status.score == 80
    assert result.decision.decision == "review"
    assert result.decision.reason_codes == ["product_status_unclear"]


def test_valid_product_rejects_claim_after_gates_pass():
    obs = {
        "product_status": {"status": "valid", "score": 0.85},
        "authenticity": {
            "mobile_capture_score": 0.9,
            "ai_generated": {"ai_probability": 0.02},
        },
        "product_match": {"score": 0.95},
        "query_match": {"score": 0.84},
    }

    result = build_link_evaluation_response(
        obs,
        AIGeneratedCheck(
            is_ai_generated=False,
            ai_probability=0.02,
            source="internal",
            signals=[],
        ),
        WebProvenanceResult(checked=True),
    )

    assert result.product_status.status == "valid"
    assert result.decision.decision == "reject_claim"
    assert result.decision.reason_codes == ["product_valid"]


def test_issue_status_with_mismatched_query_goes_to_review():
    obs = {
        "product_status": {"status": "expired", "score": 0.88},
        "authenticity": {
            "mobile_capture_score": 0.9,
            "ai_generated": {"ai_probability": 0.02},
        },
        "product_match": {"score": 0.95},
        "query_match": {"score": 0.35},
    }

    result = build_link_evaluation_response(
        obs,
        AIGeneratedCheck(
            is_ai_generated=False,
            ai_probability=0.02,
            source="internal",
            signals=[],
        ),
        WebProvenanceResult(checked=True),
    )

    assert result.product_status.status == "expired"
    assert result.decision.decision == "review"
    assert result.decision.reason_codes == ["query_match_below_threshold"]


def test_labeled_expiry_date_overrides_model_valid_status():
    obs = {
        "recognition": {
            "user_image_text": "MFG 19.02.2024 EXPIRY DATE 18.02.2026",
        },
        "product_status": {
            "status": "valid",
            "score": 0.9,
            "verdict": "The product is valid.",
            "detail": "The expiry date is 18.02.2026 and no damage is visible.",
        },
        "authenticity": {
            "mobile_capture_score": 0.9,
            "ai_generated": {"ai_probability": 0.02},
        },
        "product_match": {"score": 0.95},
        "query_match": {
            "score": 0.2,
            "query_type": "expired",
            "verdict": "The query is not supported.",
        },
    }

    result = build_link_evaluation_response(
        obs,
        AIGeneratedCheck(
            is_ai_generated=False,
            ai_probability=0.02,
            source="internal",
            signals=[],
        ),
        WebProvenanceResult(checked=True),
        query="expired product",
        today=date(2026, 6, 25),
    )

    assert result.product_status.status == "expired"
    assert result.product_status.score == 95
    assert "2026-02-18" in (result.product_status.detail or "")
    assert result.query_match.score == 95
    assert result.decision.decision == "accept_claim"
    assert result.decision.reason_codes == ["expired_claim_supported"]


def test_manufacture_date_without_expiry_label_does_not_override_valid_status():
    obs = {
        "recognition": {"user_image_text": "MFG 18.02.2026 BATCH A1"},
        "product_status": {"status": "valid", "score": 0.86},
        "authenticity": {
            "mobile_capture_score": 0.9,
            "ai_generated": {"ai_probability": 0.02},
        },
        "product_match": {"score": 0.95},
        "query_match": {"score": 0.8, "query_type": "expired"},
    }

    result = build_link_evaluation_response(
        obs,
        AIGeneratedCheck(
            is_ai_generated=False,
            ai_probability=0.02,
            source="internal",
            signals=[],
        ),
        WebProvenanceResult(checked=True),
        query="expired product",
        today=date(2026, 6, 25),
    )

    assert result.product_status.status == "valid"
    assert result.decision.decision == "reject_claim"
    assert result.decision.reason_codes == ["product_valid"]
