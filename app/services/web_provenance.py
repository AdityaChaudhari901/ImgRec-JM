"""Web reverse-image-search (req 1b): detect a "website-downloaded" photo.

Calls Google Cloud Vision WEB_DETECTION and reports how widely the image
already appears on the public web. A genuine customer damage photo should not
exist on multiple unrelated sites, so full matches across several domains are a
strong fraud signal (fused downstream in authenticity_engine).

Resilient by design: missing creds, a disabled API, or any error degrade to
`checked=False` (no signal) — never an exception. The Vision client is
synchronous, so the call runs in a worker thread under a hard timeout to avoid
blocking the event loop.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_client = None  # lazily-created Vision client singleton


@dataclass
class WebProvenanceResult:
    full_match_count: int = 0
    partial_match_count: int = 0
    distinct_pages: int = 0
    distinct_domains: int = 0
    best_guess_label: Optional[str] = None
    checked: bool = False

    def to_audit(self) -> dict:
        return {
            "full_match_count": self.full_match_count,
            "partial_match_count": self.partial_match_count,
            "distinct_pages": self.distinct_pages,
            "distinct_domains": self.distinct_domains,
            "best_guess_label": self.best_guess_label,
            "checked": self.checked,
        }


def _get_vision_client():
    """Return a cached Vision client. Built on first use (and only if enabled),
    so importing this module never needs creds or the library installed."""
    global _client
    if _client is None:
        from google.cloud import vision  # lazy import

        _client = vision.ImageAnnotatorClient()
    return _client


def reset_vision_client() -> None:
    """Drop the cached client (used by tests)."""
    global _client
    _client = None


def _domain(url: str) -> Optional[str]:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001
        return None
    if not netloc:
        return None
    return netloc[4:] if netloc.startswith("www.") else netloc


def _count_distinct_domains(urls: List[str]) -> int:
    return len({d for d in (_domain(u) for u in urls) if d})


def _decode(image_base64: str) -> bytes:
    raw = image_base64.split(",", 1)[-1].strip()
    return base64.b64decode(raw + "==", validate=False)


def _blocking_detect(image_bytes: bytes) -> WebProvenanceResult:
    client = _get_vision_client()
    # Pass image as a plain dict so tests can patch _get_vision_client without
    # needing the real google-cloud-vision library present at import time.
    response = client.web_detection(image={"content": image_bytes})
    if getattr(response, "error", None) and getattr(response.error, "message", ""):
        logger.error("vision_api_error", error=response.error.message)
        return WebProvenanceResult(checked=False)

    wd = response.web_detection
    full = list(getattr(wd, "full_matching_images", []) or [])
    partial = list(getattr(wd, "partial_matching_images", []) or [])
    pages = list(getattr(wd, "pages_with_matching_images", []) or [])
    labels = list(getattr(wd, "best_guess_labels", []) or [])

    domain_urls = [p.url for p in pages]
    return WebProvenanceResult(
        full_match_count=len(full),
        partial_match_count=len(partial),
        distinct_pages=len({p.url for p in pages}),
        distinct_domains=_count_distinct_domains(domain_urls),
        best_guess_label=(labels[0].label if labels else None),
        checked=True,
    )


async def detect_web_provenance(image_base64: str) -> WebProvenanceResult:
    """Reverse-search the image on the public web. Resilient: any failure yields
    an unchecked result (no signal) rather than raising."""
    if not settings.web_provenance_enabled:
        return WebProvenanceResult(checked=False)
    try:
        image_bytes = _decode(image_base64)
    except (binascii.Error, ValueError):
        return WebProvenanceResult(checked=False)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_blocking_detect, image_bytes),
            timeout=settings.vision_timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - Vision outage must not fail the claim
        logger.error("web_provenance_failed", error=str(exc))
        return WebProvenanceResult(checked=False)
