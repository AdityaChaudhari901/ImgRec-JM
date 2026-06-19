"""Provenance-metadata signal accuracy (false-positive guards).

C2PA "Content Credentials" are carried by real cameras (Leica/Nikon/Sony) and by
ordinary edited photos (Photoshop/Lightroom), as well as by AI tools. So the mere
presence of C2PA must NOT be treated as proof of AI generation — only the specific
assertions that genuinely indicate synthesis (the C2PA `trained algorithmic media`
digitalSourceType, Google's SynthID watermark, or a named generator) should fire.
"""

import base64
import io

import pytest
from PIL import Image, PngImagePlugin

from app.services.ai_image_detector import detect_ai_generated
from app.utils.image_metadata import inspect_metadata


def _png_with_text(**text) -> str:
    """Build a tiny PNG carrying the given text chunks, as a data-URI base64."""
    img = Image.new("RGB", (8, 8), (120, 200, 100))
    meta = PngImagePlugin.PngInfo()
    for key, value in text.items():
        meta.add_text(key, value)
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=meta)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_c2pa_provenance_alone_is_not_an_ai_signal():
    img = _png_with_text(Software="c2pa Content Credentials")
    meta = inspect_metadata(img)
    assert meta["ai_metadata_suspected"] is False
    assert "c2pa" not in meta["generator_fingerprints"]


def test_trained_algorithmic_media_assertion_is_an_ai_signal():
    img = _png_with_text(Comment="c2pa digitalSourceType: trained algorithmic media")
    meta = inspect_metadata(img)
    assert meta["ai_metadata_suspected"] is True
    assert "trained algorithmic media" in meta["generator_fingerprints"]


def test_synthid_marker_is_an_ai_signal():
    img = _png_with_text(Comment="contains synthid watermark")
    meta = inspect_metadata(img)
    assert meta["ai_metadata_suspected"] is True
    assert "synthid" in meta["generator_fingerprints"]


def test_named_generator_is_still_an_ai_signal():
    img = _png_with_text(Software="Midjourney")
    meta = inspect_metadata(img)
    assert meta["ai_metadata_suspected"] is True
    assert "midjourney" in meta["generator_fingerprints"]


def test_ambiguous_common_words_do_not_false_positive():
    # Real-image metadata/captions legitimately contain these words; an AI brand
    # sharing the name must not turn a genuine photo into an AI flag.
    captions = [
        "Photographed with the Gemini camera app",   # 'gemini'
        "Reproduction of Leonardo da Vinci's work",   # 'leonardo'
        "a firefly glowing in the dark",              # 'firefly'
        "imagen de producto",                         # 'imagen' (Spanish: image)
        "magnetic flux measurement chart",            # 'flux'
    ]
    for caption in captions:
        meta = inspect_metadata(_png_with_text(Comment=caption))
        assert meta["ai_metadata_suspected"] is False, caption


@pytest.mark.asyncio
async def test_real_image_with_c2pa_not_flagged_ai_by_detector():
    # Gemini gave a clean visual read; a bare C2PA marker must not override it.
    img = _png_with_text(Software="c2pa")
    check = await detect_ai_generated(img, {"ai_probability": 0.05, "signals": []})
    assert check.is_ai_generated is False
    assert check.ai_probability < 0.5
