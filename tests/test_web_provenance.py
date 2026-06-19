# tests/test_web_provenance.py
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.services import web_provenance
from app.services.web_provenance import (
    WebProvenanceResult,
    _count_distinct_domains,
    detect_web_provenance,
    reset_vision_client,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_vision_client()
    yield
    reset_vision_client()


def test_count_distinct_domains_strips_www_and_dedupes():
    urls = [
        "https://www.example.com/a.jpg",
        "https://example.com/b.jpg",          # same domain as above
        "http://shop.other.com/x",
        "not a url",                            # ignored
    ]
    assert _count_distinct_domains(urls) == 2


def _fake_web_detection(full=0, partial=0, pages=None, label="a shoe"):
    pages = pages or []
    wd = SimpleNamespace(
        full_matching_images=[SimpleNamespace(url=f"https://d{i}.com/i.jpg") for i in range(full)],
        partial_matching_images=[SimpleNamespace(url=f"https://p{i}.com/i.jpg") for i in range(partial)],
        pages_with_matching_images=[SimpleNamespace(url=u) for u in pages],
        best_guess_labels=[SimpleNamespace(label=label)] if label else [],
    )
    return SimpleNamespace(web_detection=wd, error=SimpleNamespace(message=""))


@pytest.mark.asyncio
async def test_full_matches_on_multiple_domains():
    fake = _fake_web_detection(
        full=2, pages=["https://a.com/p", "https://b.com/p"], label="cracked phone"
    )
    with patch.object(web_provenance, "_get_vision_client") as gc:
        gc.return_value.web_detection.return_value = fake
        result = await detect_web_provenance("data:image/jpeg;base64,/9j/fake")
    assert result.checked is True
    assert result.full_match_count == 2
    assert result.distinct_domains == 2
    assert result.best_guess_label == "cracked phone"


@pytest.mark.asyncio
async def test_clean_image_has_no_matches():
    with patch.object(web_provenance, "_get_vision_client") as gc:
        gc.return_value.web_detection.return_value = _fake_web_detection()
        result = await detect_web_provenance("data:image/jpeg;base64,/9j/fake")
    assert result.checked is True
    assert result.full_match_count == 0
    assert result.distinct_domains == 0


@pytest.mark.asyncio
async def test_vision_error_degrades_to_unchecked():
    with patch.object(web_provenance, "_get_vision_client", side_effect=RuntimeError("boom")):
        result = await detect_web_provenance("data:image/jpeg;base64,/9j/fake")
    assert result.checked is False
    assert result.full_match_count == 0


@pytest.mark.asyncio
async def test_bad_base64_degrades_to_unchecked():
    with patch.object(web_provenance, "_get_vision_client") as gc:
        result = await detect_web_provenance("not-base64-@@@")
    assert isinstance(result, WebProvenanceResult)
    assert result.checked is False
    gc.assert_not_called()  # rejected at decode, before any Vision call


@pytest.mark.asyncio
async def test_vision_error_message_degrades_to_unchecked():
    fake = _fake_web_detection(full=2, pages=["https://a.com/p"])
    fake.error = SimpleNamespace(message="RESOURCE_EXHAUSTED: quota")
    with patch.object(web_provenance, "_get_vision_client") as gc:
        gc.return_value.web_detection.return_value = fake
        result = await detect_web_provenance("data:image/jpeg;base64,/9j/fake")
    assert result.checked is False
    assert result.full_match_count == 0
