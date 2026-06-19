"""Gemini multimodal analysis for a customer damage claim.

One call produces the *observations* the scoring engine needs: image↔comment
alignment, product match against the claimed product, a visual AI-generated
hint, and any other fraud signals. The final authenticity score is NOT taken
from the model — it is computed in authenticity_engine.py.
"""

import asyncio
import base64
import binascii
import json
from typing import Optional

from google.genai import types

from app.config.settings import settings
from app.services.gemini_service import build_generation_config, get_client
from app.utils.image_utils import extract_base64_data
from app.utils.logger import get_logger

logger = get_logger(__name__)

CLAIM_PROMPT = """
You are a fraud-analysis vision AI for JioMart customer support. A customer has
raised a damage/destroyed-product claim. You are given the customer's photo, the
product the ticket is about, and the customer's comment.

CLAIMED PRODUCT: {claimed_product}
CUSTOMER COMMENT: "{user_comment}"

Analyse ONLY what is visible and return observations (do NOT make the final
approval decision — that is done downstream). Assess:

0. RECOGNITION: Describe what the photo shows in one line, list the main
   objects/products visible, and perform OCR — transcribe ALL text visible in
   the image (brand, product name, packaging text, batch/dates, etc.). If there
   is no text, use an empty string.

1. AI-GENERATED: Does the image look AI-generated / synthetic / edited rather
   than a real phone photo of a real product? Look for: impossible textures,
   warped text, inconsistent lighting/shadows, plastic-perfect surfaces,
   nonsensical label text, GAN artifacts. Report `ai_probability` where
   0.0 = clearly a real photo and 1.0 = clearly AI-generated/edited.
2. ALIGNMENT: How well does the image actually show the problem the customer
   describes in their comment? (e.g. comment says "leaking" — is leakage visible?)
3. PRODUCT MATCH: Is the product in the image the same as, or a plausible match
   for, the CLAIMED PRODUCT above?
4. OTHER FLAGS: Anything else suspicious — looks like a stock/marketing photo,
   a screenshot, a different unrelated item, a photo of a screen, watermarks, etc.

RESPOND ONLY WITH THIS EXACT JSON — NO PREAMBLE, NO MARKDOWN:

{{
  "recognition": {{
    "scene": "one-line description of what the photo shows",
    "objects": ["main object/product", "..."],
    "extracted_text": "all text visible in the image, or empty string"
  }},
  "ai_generated": {{
    "ai_probability": 0.0,
    "signals": ["short reason", "..."]
  }},
  "image_comment_alignment": {{
    "score": 0.0,
    "aligned": true,
    "reason": "one sentence"
  }},
  "product_match": {{
    "detected_product": "what you see in the image",
    "matches": true,
    "score": 0.0,
    "reason": "one sentence"
  }},
  "other_flags": ["short flag", "..."],
  "summary": "one sentence analyst summary"
}}
"""


async def analyze_claim(
    image_base64: str,
    user_comment: str,
    claimed_product: str,
    reference_image_base64: Optional[str] = None,
) -> dict:
    """Run the Gemini claim analysis and return parsed observations JSON."""
    raw_b64, mime_type = extract_base64_data(image_base64)
    try:
        image_bytes = base64.b64decode(raw_b64 + "==", validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc

    prompt = CLAIM_PROMPT.format(
        claimed_product=claimed_product.strip(),
        user_comment=user_comment.strip().replace('"', "'"),
    )

    contents: list = [types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
    if reference_image_base64:
        try:
            ref_b64, ref_mime = extract_base64_data(reference_image_base64)
            ref_bytes = base64.b64decode(ref_b64 + "==", validate=False)
            contents.append(types.Part.from_bytes(data=ref_bytes, mime_type=ref_mime))
            prompt += "\n(The SECOND image is the official reference product photo.)"
        except Exception:  # noqa: BLE001 - reference is optional
            pass
    contents.append(prompt)

    client = get_client()
    config = build_generation_config()

    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=settings.gemini_model, contents=contents, config=config
            ),
            timeout=settings.gemini_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Gemini API timed out after {settings.gemini_timeout_seconds}s"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("claim_gemini_failed", error=str(exc))
        raise

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("claim_bad_json", error=str(exc), raw=text[:500])
        raise ValueError("Gemini returned malformed JSON") from exc
