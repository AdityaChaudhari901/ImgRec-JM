import pytest

from app.services.ai_image_detector import detect_ai_generated


@pytest.mark.asyncio
async def test_missing_ai_hint_is_not_treated_as_generated():
    check = await detect_ai_generated("data:image/jpeg;base64,/9j/fake", {})

    assert check.is_ai_generated is False
    assert check.ai_probability == 0.0
    assert check.source == "internal"
