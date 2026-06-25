import asyncio
import json
from unittest.mock import patch

import pytest

from app.services.dispute_service import analyze_dispute


class _Resp:
    def __init__(self, text):
        self.text = text


def test_analyze_dispute_parses_json():
    payload = {"ocr": {"printed_mrp_values": [90.0]}, "product_match": {"matches": False}}

    async def fake_gen(**kwargs):
        return _Resp(json.dumps(payload))

    with patch("app.services.dispute_service.generate_content_with_fallback", side_effect=fake_gen):
        out = asyncio.get_event_loop().run_until_complete(
            analyze_dispute(["data:image/jpeg;base64,AAAA"], "mrp_abuse", "Oil 1L", "overcharged")
        )
    assert out["product_match"]["matches"] is False


def test_analyze_dispute_sends_all_images_capped(monkeypatch):
    from app.config.settings import settings
    monkeypatch.setattr(settings, "dispute_max_images", 2)
    captured = {}

    async def fake_gen(**kwargs):
        captured["contents"] = kwargs["contents"]
        return _Resp(json.dumps({"ocr": {}, "ai_generated": {}, "summary": "ok"}))

    imgs = ["data:image/jpeg;base64,AAAA"] * 5  # 5 supplied, cap is 2
    with patch("app.services.dispute_service.generate_content_with_fallback", side_effect=fake_gen):
        asyncio.get_event_loop().run_until_complete(
            analyze_dispute(imgs, "damaged", "Product", "leaking")
        )
    # contents = capped image parts (2) + 1 prompt string
    assert len(captured["contents"]) == 3


def test_analyze_dispute_empty_text_raises():
    async def fake_gen(**kwargs):
        return _Resp("")

    with patch("app.services.dispute_service.generate_content_with_fallback", side_effect=fake_gen):
        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(
                analyze_dispute(["data:image/jpeg;base64,AAAA"], None, "x", "y")
            )
