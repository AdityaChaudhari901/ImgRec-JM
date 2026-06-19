"""Perceptual-hash (dHash) tests.

Phase 1 only needs: a stable, deterministic hash per image, distinct hashes for
distinct images, and a safe `None` for undecodable bytes (so the idempotency key
falls back to an exact-bytes hash). The near-duplicate / Hamming-distance lookup
is Phase 2.
"""

import base64
import io

from PIL import Image

from app.utils.image_utils import compute_image_phash


def _gradient_b64(seed: int, size: int = 48) -> str:
    """A deterministic grayscale gradient (solid colours hash to all-zero dHash)."""
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (x * seed + y * 3) % 256
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_phash_is_deterministic_and_64_bit():
    b64 = _gradient_b64(7)
    h1 = compute_image_phash(b64)
    h2 = compute_image_phash(b64)
    assert h1 == h2
    assert h1 is not None
    assert len(h1) == 16  # 64 bits as hex


def test_phash_differs_for_different_images():
    assert compute_image_phash(_gradient_b64(7)) != compute_image_phash(_gradient_b64(31))


def test_phash_returns_none_for_undecodable_bytes():
    assert compute_image_phash("data:image/jpeg;base64,/9j/not-a-real-image") is None
