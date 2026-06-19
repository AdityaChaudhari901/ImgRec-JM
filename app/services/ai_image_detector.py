"""Pluggable AI-generated-image detection.

Two providers behind one interface so we can upgrade without touching callers:

- "internal"   : free, in-process — ensembles image metadata fingerprints
                 (image_metadata.inspect_metadata) with Gemini's visual hint.
                 ~$0 extra, advisory-grade.
- "sightengine": calls the paid specialist detector (~$0.01/image) and still
                 folds in the free metadata signal. Higher accuracy.

Returns an AIGeneratedCheck. The result is *advisory* — the scoring engine
decides how much it influences routing (it never hard-rejects on the internal
provider alone).
"""

import base64
import binascii
from typing import Optional

from app.config.settings import settings
from app.models.verify_response import AIGeneratedCheck
from app.utils.image_metadata import inspect_metadata
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SIGHTENGINE_URL = "https://api.sightengine.com/1.0/check.json"


async def detect_ai_generated(
    image_base64: str, gemini_hint: Optional[dict] = None
) -> AIGeneratedCheck:
    """Decide whether the image looks AI-generated, using the configured provider."""
    meta = inspect_metadata(image_base64)
    if settings.ai_detector_provider == "sightengine" and settings.sightengine_api_user:
        return await _sightengine(image_base64, meta)
    return _internal(meta, gemini_hint or {})


def _hint_probability(gemini_hint: dict) -> float:
    """Normalise the model's AI hint to P(AI-generated) in [0,1].

    Prefers the new `ai_probability` field; falls back to the old
    is_ai_generated + confidence shape (confidence-in-boolean) for safety.
    """
    if not gemini_hint:
        return 0.0

    if "ai_probability" in gemini_hint and gemini_hint["ai_probability"] is not None:
        try:
            return max(0.0, min(1.0, float(gemini_hint["ai_probability"])))
        except (TypeError, ValueError):
            return 0.0

    if "is_ai_generated" not in gemini_hint:
        return 0.0

    if "confidence" not in gemini_hint:
        return 0.5 if gemini_hint.get("is_ai_generated") else 0.0

    try:
        conf = max(0.0, min(1.0, float(gemini_hint.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        return 0.0
    return conf if gemini_hint.get("is_ai_generated") else 1.0 - conf


def _internal(meta: dict, gemini_hint: dict) -> AIGeneratedCheck:
    signals: list[str] = []

    # Start from Gemini's visual judgement (advisory), as P(AI).
    p_ai = _hint_probability(gemini_hint)
    for s in gemini_hint.get("signals", []) or []:
        signals.append(f"visual: {s}")

    # Metadata fingerprint is a stronger, cheap signal — it pushes P(AI) up.
    if meta.get("ai_metadata_suspected"):
        p_ai = max(p_ai, 0.85)
        fps = ", ".join(meta.get("generator_fingerprints", []))
        signals.append(f"metadata fingerprint: {fps}")

    # A real camera EXIF is evidence *against* AI; soften a weak visual guess.
    if meta.get("has_camera_exif") and not meta.get("ai_metadata_suspected"):
        signals.append(f"camera EXIF present: {meta.get('camera') or 'unknown device'}")
        if p_ai < 0.7:
            p_ai = max(0.0, p_ai - 0.2)

    p_ai = round(min(p_ai, 1.0), 3)
    return AIGeneratedCheck(
        is_ai_generated=p_ai >= 0.5,
        ai_probability=p_ai,
        source="internal",
        signals=signals,
    )


async def _sightengine(image_base64: str, meta: dict) -> AIGeneratedCheck:
    try:
        import httpx
    except Exception:  # noqa: BLE001
        logger.error("sightengine_httpx_missing")
        return _internal(meta, {})

    try:
        raw = image_base64.split(",", 1)[-1].strip()
        img_bytes = base64.b64decode(raw + "==", validate=False)
    except (binascii.Error, ValueError):
        return _internal(meta, {})

    data = {
        "models": "genai",
        "api_user": settings.sightengine_api_user,
        "api_secret": settings.sightengine_api_secret,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.gemini_timeout_seconds) as client:
            resp = await client.post(
                _SIGHTENGINE_URL, data=data, files={"media": ("claim.jpg", img_bytes)}
            )
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully to internal
        logger.error("sightengine_call_failed", error=str(exc))
        return _internal(meta, {})

    prob = float(payload.get("type", {}).get("ai_generated", 0.0) or 0.0)
    signals = [f"sightengine ai_generated probability: {prob:.2f}"]
    if meta.get("ai_metadata_suspected"):
        prob = max(prob, 0.85)
        signals.append("metadata fingerprint: " + ", ".join(meta["generator_fingerprints"]))

    prob = round(min(prob, 1.0), 3)
    return AIGeneratedCheck(
        is_ai_generated=prob >= 0.5,
        ai_probability=prob,
        source="sightengine",
        signals=signals,
    )
