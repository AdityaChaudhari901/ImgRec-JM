import asyncio
import base64
import binascii
import json
from datetime import date

from google import genai
from google.genai import types

from app.config.settings import settings
from app.utils.gcp_auth import google_sa_credentials
from app.utils.image_utils import extract_base64_data
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazily-created singleton clients. Built on first use so importing this module
# (e.g. during tests) never needs credentials or network.
_clients: dict[str, genai.Client] = {}


def build_generation_config(max_output_tokens: int = 2048) -> types.GenerateContentConfig:
    """Shared JSON-mode generation config.

    `thinking_budget=0` disables Gemini 2.5's internal "thinking" — otherwise it
    silently consumes the output-token budget and truncates the JSON. We don't
    need chain-of-thought for structured extraction, and disabling it is faster
    and cheaper. Guarded so it degrades gracefully on SDK/model variants that
    don't support ThinkingConfig.
    """
    kwargs = dict(
        temperature=0.1,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
    )
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:  # noqa: BLE001
        pass
    return types.GenerateContentConfig(**kwargs)


def build_claim_generation_config(
    max_output_tokens: int = 2048,
) -> types.GenerateContentConfig:
    """Generation config for the claim analysis, with a response_schema so the
    JSON shape can't drift or come back malformed. Mirrors build_generation_config
    but pins the structure analyze_claim depends on."""
    claim_schema = {
        "type": "object",
        "properties": {
            "recognition": {
                "type": "object",
                "properties": {
                    "scene": {"type": "string"},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "extracted_text": {"type": "string"},
                },
            },
            "ai_generated": {
                "type": "object",
                "properties": {
                    "ai_probability": {"type": "number"},
                    "signals": {"type": "array", "items": {"type": "string"}},
                },
            },
            "image_comment_alignment": {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "aligned": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
            "product_match": {
                "type": "object",
                "properties": {
                    "detected_product": {"type": "string"},
                    "matches": {"type": "boolean"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
            "other_flags": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": [
            "recognition",
            "ai_generated",
            "image_comment_alignment",
            "product_match",
            "summary",
        ],
    }
    kwargs = dict(
        temperature=0.1,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=claim_schema,
    )
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:  # noqa: BLE001
        pass
    return types.GenerateContentConfig(**kwargs)


def build_dispute_generation_config(max_output_tokens: int = 2048) -> types.GenerateContentConfig:
    """JSON-mode config pinning the dispute observation schema."""
    schema = {
        "type": "object",
        "properties": {
            "recognition": {"type": "object", "properties": {
                "scene": {"type": "string"},
                "objects": {"type": "array", "items": {"type": "string"}},
            }},
            "ocr": {"type": "object", "properties": {
                "raw_text": {"type": "string"},
                "printed_mrp_values": {"type": "array", "items": {"type": "number"}},
                "manufacture_date": {"type": "string"},
                "expiry_date": {"type": "string"},
                "batch_no": {"type": "string"},
            }},
            "product_match": {"type": "object", "properties": {
                "detected_product": {"type": "string"},
                "matches": {"type": "boolean"},
                "score": {"type": "number"},
                "reason": {"type": "string"},
            }},
            "damage": {"type": "object", "properties": {
                "detected": {"type": "boolean"},
                "type": {"type": "string"},
                "severity": {"type": "string"},
                "description": {"type": "string"},
            }},
            "quality": {"type": "object", "properties": {
                "poor_quality": {"type": "boolean"},
                "indicators": {"type": "array", "items": {"type": "string"}},
                "supports_claim": {"type": "boolean"},
                "internal_defect": {"type": "boolean"},
            }},
            "spoilage": {"type": "object", "properties": {
                "mold_or_visible_spoilage": {"type": "boolean"},
            }},
            "count": {"type": "object", "properties": {
                "counted_units": {"type": "integer"},
                "confidence": {"type": "number"},
            }},
            "counterfeit_suspected": {"type": "boolean"},
            "ai_generated": {"type": "object", "properties": {
                "ai_probability": {"type": "number"},
                "signals": {"type": "array", "items": {"type": "string"}},
            }},
            "summary": {"type": "string"},
        },
        "required": ["ocr", "ai_generated", "summary"],
    }
    kwargs = dict(temperature=0.1, max_output_tokens=max_output_tokens,
                  response_mime_type="application/json", response_schema=schema)
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:  # noqa: BLE001
        pass
    return types.GenerateContentConfig(**kwargs)


def get_client(provider: str | None = None) -> genai.Client:
    """Return a cached google-genai client for the configured provider.

    - Vertex AI  -> bills the GCP project, auth via Application Default Creds.
    - AI Studio  -> uses the GOOGLE_API_KEY and its prepay billing.
    """
    selected = provider or ("vertex" if settings.use_vertex else "api_key")
    if selected not in _clients:
        if selected == "vertex":
            _clients[selected] = genai.Client(
                vertexai=True,
                project=settings.vertex_project_id,
                location=settings.vertex_region,
                credentials=google_sa_credentials(),
            )
        else:
            _clients[selected] = genai.Client(api_key=settings.google_api_key)
    return _clients[selected]


async def generate_content_with_fallback(
    *,
    model: str,
    contents,
    config: types.GenerateContentConfig,
):
    """Generate content with the configured Gemini provider.

    If AI Studio is the primary provider and it returns a quota/prepay 429,
    retry through Vertex AI when the deployment has a Vertex project configured.
    This lets Boltic use GCP trial credits even if the API-key project has no
    prepay balance.
    """
    primary = "vertex" if settings.use_vertex else "api_key"
    try:
        return await get_client(primary).aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        if not _should_retry_with_vertex(exc, primary):
            raise
        logger.warning(
            "gemini_retrying_with_vertex",
            primary_provider=primary,
            code=getattr(exc, "code", None),
        )
        try:
            return await get_client("vertex").aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as fallback_exc:  # noqa: BLE001
            logger.error(
                "gemini_vertex_fallback_failed",
                code=getattr(fallback_exc, "code", None),
                error=str(fallback_exc),
            )
            raise


def _should_retry_with_vertex(exc: Exception, primary: str) -> bool:
    if primary == "vertex" or not settings.gemini_vertex_fallback_enabled:
        return False
    if settings.vertex_project_id in {"", "missing-project-id"}:
        return False
    code = getattr(exc, "code", None)
    if code != 429:
        return False
    text = str(exc).lower()
    return (
        "resource_exhausted" in text
        or "prepayment credits are depleted" in text
        or "quota" in text
    )


SYSTEM_PROMPT = """
You are a product inspection AI for JioMart's customer support system.
Your job is to analyze a product image and return a structured JSON report.

WHAT TO LOOK FOR:

1. TEXT ON LABEL (OCR):
   - Manufacture date — labeled as: MFG, Mfd, Manufactured, Mfg Date, Packed On
   - Expiry date — labeled as: EXP, Expiry, Best Before, Use By, BB, Expires
   - Batch / Lot number — labeled as: Batch, Lot, Lot No, Batch No
   - Extract all visible label text even if dates are unclear

2. PHYSICAL DAMAGE:
   - Crushed or deformed packaging
   - Tears, cuts, or holes
   - Broken or missing seals
   - Liquid leakage or staining
   - Severe dents (cans/bottles)
   - Discoloration or mold

3. EVIDENCE AUTHENTICITY:
   - Decide whether the image looks AI-generated, synthetic, deepfake, or
     digitally fabricated instead of being a real phone photo of a real product.
   - Look for impossible packaging geometry, warped or nonsensical text,
     plastic-perfect liquid/surfaces, inconsistent shadows/reflections,
     generated-looking hands/backgrounds, and image-tool artifacts.
   - If evidence looks synthetic, report a high ai_probability. Do not let
     synthetic-looking damage trigger a refund or exchange; downstream code will
     block automated action and route to review.

DATE PARSING RULES:
   - Handle all Indian formats: DD/MM/YYYY, MM/YYYY, MMM YYYY, DD-MM-YY, MON-YY
   - Convert all output dates to ISO format: YYYY-MM-DD
   - If only month/year found (e.g. "JUN 2025"), use last day of that month
   - Today's date is: {today}

STATUS LOGIC:
   - "expired"  -> expiry date is before today
   - "damaged"  -> physical damage detected AND product not expired
   - "valid"    -> not expired AND no damage
   - "unclear"  -> cannot read dates AND no visible damage

ACTION LOGIC:
   - expired                -> initiate_refund,   refund_eligible: true,  priority: high
   - damaged + severe       -> initiate_refund,   refund_eligible: true,  priority: high
   - damaged + moderate     -> initiate_exchange, refund_eligible: false, priority: medium
   - damaged + minor        -> initiate_exchange, refund_eligible: false, priority: low
   - valid                  -> no_action,         refund_eligible: false, priority: low
   - unclear                -> no_action,         refund_eligible: false, priority: low

RESPOND ONLY WITH THIS EXACT JSON — NO PREAMBLE, NO MARKDOWN, NO EXPLANATION:

{{
  "status": "expired | damaged | valid | unclear",
  "confidence": 0.0,
  "ocr": {{
    "manufacture_date": "YYYY-MM-DD or null",
    "expiry_date": "YYYY-MM-DD or null",
    "batch_no": "string or null",
    "raw_text": "all visible label text as single string"
  }},
  "damage": {{
    "detected": true,
    "type": "crushed_packaging | tear | broken_seal | leakage | dent | discoloration | mold | null",
    "severity": "minor | moderate | severe | null",
    "description": "one sentence description or null"
  }},
  "ai_generated": {{
    "ai_probability": 0.0,
    "signals": ["short reason", "..."]
  }},
  "action": {{
    "type": "initiate_refund | initiate_exchange | no_action",
    "message": "customer-facing explanation in one sentence",
    "refund_eligible": true,
    "priority": "high | medium | low"
  }}
}}
"""

# scan_type lets Kaily bias the model toward one analysis path.
_SCAN_HINTS = {
    "ocr": "\nFOCUS: Prioritise reading dates and label text. Damage is secondary.\n",
    "damage": "\nFOCUS: Prioritise physical damage assessment. OCR is secondary.\n",
    "auto": "",
}


async def analyze_image(image_base64: str, scan_type: str = "auto") -> dict:
    """Call Gemini with the product image and return the parsed JSON report."""
    raw_b64, mime_type = extract_base64_data(image_base64)
    try:
        image_bytes = base64.b64decode(raw_b64 + "==", validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc

    today = date.today().isoformat()
    prompt = SYSTEM_PROMPT.format(today=today) + _SCAN_HINTS.get(scan_type, "")

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    config = build_generation_config()

    try:
        response = await asyncio.wait_for(
            generate_content_with_fallback(
                model=settings.gemini_model,
                contents=[image_part, prompt],
                config=config,
            ),
            timeout=settings.gemini_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Gemini API timed out after {settings.gemini_timeout_seconds}s"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - re-raised after logging
        logger.error("gemini_call_failed", error=str(exc))
        raise

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("gemini_bad_json", error=str(exc), raw=text[:500])
        raise ValueError("Gemini returned malformed JSON") from exc
