"""Gemini multi-image observation pass for a grocery dispute.

Returns visible observations only (OCR, product match, damage, quality,
spoilage, unit count, counterfeit/AI hints). The deterministic decision is made
downstream in dispute_engine.py.
"""

import asyncio
import base64
import binascii
import json
from typing import List, Optional

from google.genai import types

from app.config.settings import settings
from app.services.gemini_service import (
    build_dispute_generation_config,
    generate_content_with_fallback,
)
from app.utils.image_utils import extract_base64_data
from app.utils.logger import get_logger

logger = get_logger(__name__)

DISPUTE_PROMPT = """
You are a product-inspection vision AI for JioMart customer support. A customer
raised a delivery dispute. You are given 1 or more customer photos, the ordered
product name, and the customer's comment. Report ONLY what is visible — do NOT
make the refund decision (that is done downstream).

ORDERED PRODUCT: {product_name}
DISPUTE CATEGORY (may be empty): {category}
CUSTOMER COMMENT: "{comment}"

Assess and return:
- OCR: transcribe ALL label text (raw_text). List every MRP/price value you can
  read as printed_mrp_values (numbers only; include both struck-through and final
  prices if present). Extract manufacture_date and expiry_date in ISO YYYY-MM-DD
  if readable (use last day of month if only month/year), else "".
- PRODUCT MATCH: does the product in the image match the ORDERED PRODUCT? matches=true/false.
- DAMAGE: detected true/false; type one of crushed_packaging|tear|broken_seal|
  leakage|dent|discoloration|mold|tamper|resealed|missing_component; severity
  minor|moderate|severe.
- QUALITY: poor_quality true/false (discoloration, wilting, surface damage,
  visible defect); supports_claim true/false; internal_defect true/false (a defect
  that needs warranty/performance testing, not visible spoilage).
- SPOILAGE: mold_or_visible_spoilage true/false.
- COUNT: counted_units = number of distinct retail units visible (integer);
  confidence 0..1.
- counterfeit_suspected true/false.
- AI-GENERATED: ai_probability 0..1 (0 real photo, 1 synthetic).

RESPOND ONLY WITH JSON matching the provided schema. No preamble, no markdown.
"""


def _to_part(image_base64: str) -> types.Part:
    raw_b64, mime_type = extract_base64_data(image_base64)
    try:
        image_bytes = base64.b64decode(raw_b64 + "==", validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image is not valid base64") from exc
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


async def analyze_dispute(
    images: List[str], category: Optional[str], product_name: str, comment: str
) -> dict:
    parts: list = [_to_part(img) for img in images[: settings.dispute_max_images]]
    prompt = DISPUTE_PROMPT.format(
        product_name=(product_name or "").strip(),
        category=(category or "").strip(),
        comment=(comment or "").strip().replace('"', "'"),
    )
    contents = parts + [prompt]
    config = build_dispute_generation_config()

    try:
        response = await asyncio.wait_for(
            generate_content_with_fallback(
                model=settings.gemini_model, contents=contents, config=config
            ),
            timeout=settings.gemini_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Gemini API timed out after {settings.gemini_timeout_seconds}s"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("dispute_gemini_failed", error=str(exc))
        raise

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("dispute_bad_json", error=str(exc), raw=text[:500])
        raise ValueError("Gemini returned malformed JSON") from exc
