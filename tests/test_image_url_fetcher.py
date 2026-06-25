import socket
from io import BytesIO

import pytest
from PIL import Image

from app.config.settings import settings
from app.services.image_url_fetcher import (
    FetchedImage,
    ImageUrlError,
    _optimize_for_model,
    pixelbin_url_for,
    validate_public_http_url,
)


@pytest.mark.asyncio
async def test_validate_public_http_url_rejects_private_ip():
    with pytest.raises(ImageUrlError) as exc:
        await validate_public_http_url("http://127.0.0.1/private.jpg", role="user image")

    assert "non-public" in exc.value.detail


@pytest.mark.asyncio
async def test_validate_public_http_url_rejects_private_dns(monkeypatch):
    def fake_getaddrinfo(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(ImageUrlError) as exc:
        await validate_public_http_url("https://example.com/image.jpg", role="product image")

    assert "non-public" in exc.value.detail


@pytest.mark.asyncio
async def test_validate_public_http_url_accepts_public_dns(monkeypatch):
    def fake_getaddrinfo(*_args, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    assert (
        await validate_public_http_url("https://example.com/image.jpg", role="product image")
    ) == "https://example.com/image.jpg"


def test_pixelbin_url_template_encodes_source_url(monkeypatch):
    monkeypatch.setattr(settings, "pixelbin_enabled", True)
    monkeypatch.setattr(
        settings,
        "pixelbin_url_template",
        "https://cdn.pixelbin.io/v2/demo/url/wrkr/t.resize(w:1280)/{url_encoded}",
    )

    result = pixelbin_url_for("https://shop.example.com/products/a b.jpg?x=1&y=2")

    assert result.startswith("https://cdn.pixelbin.io/v2/demo/url/wrkr/")
    assert "https%3A%2F%2Fshop.example.com%2Fproducts%2Fa%20b.jpg%3Fx%3D1%26y%3D2" in result


def test_fetched_image_uses_original_when_no_model_variant():
    data = b"original"
    image = FetchedImage(
        source_url="https://example.com/a.jpg",
        fetched_url="https://example.com/a.jpg",
        mime_type="image/jpeg",
        data=data,
    )

    assert image.model_bytes == data
    assert image.model_content_type == "image/jpeg"
    assert image.model_size_bytes == len(data)


def test_optimize_for_model_resizes_large_image(monkeypatch):
    monkeypatch.setattr(settings, "link_eval_model_max_edge_px", 128)
    monkeypatch.setattr(settings, "link_eval_model_image_quality", 85)
    original = BytesIO()
    Image.new("RGB", (900, 700), color="red").save(original, format="PNG")

    mime_type, optimized = _optimize_for_model(
        original.getvalue(),
        "image/png",
        "user image",
    )

    assert mime_type == "image/jpeg"
    assert optimized is not None
    assert len(optimized) < len(original.getvalue())
    with Image.open(BytesIO(optimized)) as image:
        assert max(image.size) <= 128
