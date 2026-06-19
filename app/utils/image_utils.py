import base64
import binascii
import io
from typing import Optional

from PIL import Image

from app.config.settings import settings

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Magic-byte signatures used as a fallback when no data-URI header is present.
_SIGNATURES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF", "image/webp"),  # WebP files start with "RIFF....WEBP"
)


def extract_base64_data(image_base64: str) -> tuple[str, str]:
    """
    Accepts a full data URI or a raw base64 string.
    Returns (raw_base64_string, mime_type).
    """
    if image_base64.startswith("data:"):
        header, _, data = image_base64.partition(",")
        mime_type = header.split(";")[0].replace("data:", "") or "image/jpeg"
    else:
        data = image_base64
        mime_type = detect_mime_type(image_base64)

    data = data.strip()

    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Unsupported image type: {mime_type}. Use JPEG, PNG, or WebP.")

    return data, mime_type


def detect_mime_type(image_base64: str) -> str:
    """
    Best-effort mime detection from a raw (headerless) base64 string by decoding
    the first few bytes and matching known image signatures. Defaults to JPEG.
    """
    raw = image_base64.split(",", 1)[-1].strip()
    try:
        prefix = base64.b64decode(raw[:32] + "===")
    except (binascii.Error, ValueError):
        return "image/jpeg"

    for signature, mime in _SIGNATURES:
        if prefix.startswith(signature):
            return mime
    return "image/jpeg"


def validate_image_size(image_base64: str) -> None:
    """Reject images whose decoded size exceeds MAX_IMAGE_SIZE_MB."""
    raw = image_base64.split(",", 1)[-1].strip()
    try:
        size_bytes = len(base64.b64decode(raw + "==", validate=False))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc

    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValueError(f"Image exceeds {settings.max_image_size_mb}MB limit")


def _dhash_hex(img: Image.Image, hash_size: int = 8) -> str:
    """Difference-hash: resize to (hash_size+1 x hash_size) grayscale and emit one
    bit per horizontal neighbour comparison. dHash (not pHash) so we avoid a scipy
    dependency while staying robust to recompression/scaling — good enough for the
    Phase 2 Hamming-distance dedup that builds on this hex.
    """
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    px = list(small.getdata())
    width = hash_size + 1
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * width + col]
            right = px[row * width + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:0{hash_size * hash_size // 4}x}"


def compute_image_phash(image_base64: str) -> Optional[str]:
    """Best-effort perceptual hash (hex) of an image. Returns None on any failure
    (corrupt/undecodable bytes), so callers fall back to an exact-bytes idempotency
    key rather than crashing.
    """
    raw = image_base64.split(",", 1)[-1].strip()
    try:
        data = base64.b64decode(raw + "==", validate=False)
        return _dhash_hex(Image.open(io.BytesIO(data)))
    except Exception:  # noqa: BLE001 - best-effort; no phash for unprocessable input
        return None


def assert_decodable_image(image_base64: str) -> None:
    """
    Verify the bytes actually decode to a real raster image. Raises ValueError
    for unprocessable input (used to surface a 422 to the caller).
    """
    raw = image_base64.split(",", 1)[-1].strip()
    try:
        data = base64.b64decode(raw + "==", validate=False)
        Image.open(io.BytesIO(data)).verify()
    except Exception as exc:  # noqa: BLE001 - any failure means unprocessable
        raise ValueError("Image could not be decoded or is not a supported format") from exc
