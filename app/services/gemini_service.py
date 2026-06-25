import asyncio

from google import genai
from google.genai import types

from app.config.settings import settings
from app.utils.gcp_auth import google_sa_credentials
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazily-created singleton clients. Built on first use so importing this module
# (e.g. during tests) never needs credentials or network.
_clients: dict[str, genai.Client] = {}

# Per-instance backpressure: cap concurrent model calls so a traffic spike can't
# overrun the Vertex/AI-Studio quota or exhaust worker tasks. Lazily created so
# import stays side-effect-free.
_gemini_semaphore: "asyncio.Semaphore | None" = None


def _get_gemini_semaphore() -> asyncio.Semaphore:
    global _gemini_semaphore
    if _gemini_semaphore is None:
        _gemini_semaphore = asyncio.Semaphore(settings.gemini_max_concurrency)
    return _gemini_semaphore


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
    A per-instance semaphore bounds concurrent model calls (backpressure).
    """
    primary = "vertex" if settings.use_vertex else "api_key"
    async with _get_gemini_semaphore():
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
