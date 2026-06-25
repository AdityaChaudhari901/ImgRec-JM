import asyncio
import base64
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.config.settings import settings
from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.models.dispute_request import DisputeRequest
from app.models.dispute_response import DisputeResponse, RefundResult
from app.services.audit_service import build_idempotency_key, find_replay, persist_decision
from app.services.category_classifier import classify_category
from app.services.dedup_service import find_duplicates, register_image
from app.services.dispute_engine import decide
from app.services.dispute_service import analyze_dispute
from app.services.image_url_fetcher import ImageUrlError, download_image_url
from app.utils.image_utils import compute_image_phash, validate_image_size
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _autonomous() -> set:
    return {c.strip() for c in settings.dispute_autonomous_categories.split(",") if c.strip()}


def _request_id() -> str:
    return f"dsp_{int(time.time() * 1000)}"


def _order_id(body: DisputeRequest) -> str:
    if body.shipment is not None:
        return body.shipment.order_tracking_id
    return (body.claim_id or "no-order")


async def _resolve_images(body: DisputeRequest) -> list[str]:
    """Return a list of base64 data-URI images from `images` and/or `image_urls`.

    Base64 images are size-checked here; URL images are fetched through the
    SSRF-safe fetcher (which bounds size/type) and converted to a model-optimized
    data URI so large CDN originals don't risk a Gemini timeout.
    """
    images: list[str] = []
    for img in body.images or []:
        validate_image_size(img)
        images.append(img)

    if body.image_urls:
        try:
            fetched = await asyncio.gather(
                *[download_image_url(u, role="dispute image") for u in body.image_urls]
            )
        except ImageUrlError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)
        for f in fetched:
            encoded = base64.b64encode(f.model_bytes).decode("ascii")
            images.append(f"data:{f.model_content_type};base64,{encoded}")

    if not images:
        raise HTTPException(status_code=422, detail="no usable images provided")
    return images


@router.post(
    "/api/v1/imgrecog/dispute",
    response_model=DisputeResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("100/minute")
async def dispute(request: Request, body: DisputeRequest) -> DisputeResponse:
    start = time.time()
    order_id = _order_id(body)

    # Resolve all images (base64 + fetched URLs) up front.
    try:
        images = await _resolve_images(body)
    except HTTPException:
        raise
    primary_image = images[0]
    image_phash = compute_image_phash(primary_image)

    # Resolve the category BEFORE the idempotency identity so two different
    # disputes on the same order+image don't collide, and a rebuttal is distinct
    # from the original. An explicit idempotency_key/claim_id still wins.
    category, source = classify_category(body.dispute_category, body.ticket)
    idem_scope = f"dispute:{category}:rb{int(body.is_rebuttal)}"
    idempotency_key = build_idempotency_key(
        body.idempotency_key, body.claim_id, order_id, idem_scope, image_phash, primary_image,
    )
    replay = await find_replay("dispute", idempotency_key)
    if replay is not None:
        logger.info("idempotent_replay", endpoint="dispute", order_id=order_id,
                    request_id=replay.request_id)
        return replay

    if category is None:
        resp = _assemble(body, None, "none", {}, decision="agent", route="agent",
                         flags=["insufficient_data"], refund=RefundResult(),
                         recommendation="No category resolvable from description, notes, or disposition.",
                         confidence=0.0)
        return await _persist(request, body, primary_image, image_phash, idempotency_key, resp, {}, start)

    product_name = body.shipment.product_name if body.shipment is not None else ""
    try:
        observations = await analyze_dispute(
            images, category, product_name, body.ticket.description or body.ticket.title,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Image analysis timed out")
    except ValueError as exc:
        logger.warning("dispute_analysis_failed", error=str(exc), order_id=order_id)
        resp = _assemble(body, category, source, {}, decision="agent", route="agent",
                         flags=["low_confidence"], refund=RefundResult(),
                         recommendation="Image could not be analysed; route to agent.", confidence=0.0)
        return await _persist(request, body, primary_image, image_phash, idempotency_key, resp, {}, start)

    observations["_desc_len"] = len((body.ticket.description or "").strip())

    dedup_result = await find_duplicates(image_phash, order_id, "dispute")
    signals = {
        "ai_probability": float((observations.get("ai_generated") or {}).get("ai_probability", 0.0)),
        "dedup_cross": dedup_result.is_cross_claim_duplicate,
        "web_hard": False,
    }

    d = decide(category, source, observations, body.shipment, body.is_rebuttal, signals)

    route = "agent" if (
        d.decision == "agent" or settings.dispute_assist_mode or category not in _autonomous()
    ) else "auto"

    resp = _assemble(body, category, source, observations, decision=d.decision, route=route,
                     flags=d.agent_flags, refund=RefundResult(**d.refund),
                     recommendation=d.recommendation, confidence=d.confidence)
    out = await _persist(request, body, primary_image, image_phash, idempotency_key, resp,
                         observations, start)
    await register_image(image_phash, order_id, "dispute", out.request_id)
    logger.info("dispute_complete", request_id=out.request_id, order_id=order_id,
                category=category, decision=out.decision, route=out.route,
                latency_ms=round((time.time() - start) * 1000))
    return out


def _assemble(body, category, source, observations, *, decision, route, flags,
              refund, recommendation, confidence) -> DisputeResponse:
    obs = {k: v for k, v in (observations or {}).items() if not k.startswith("_")}
    return DisputeResponse(
        request_id=_request_id(), order_tracking_id=_order_id(body),
        category=category, category_source=source, decision=decision, route=route,
        agent_flags=flags, refund=refund, recommendation=recommendation,
        confidence=confidence, observations=obs, model_used=settings.gemini_model,
    )


async def _persist(request, body, primary_image, image_phash, idempotency_key, resp,
                   observations, start) -> DisputeResponse:
    latency_ms = round((time.time() - start) * 1000)
    return await persist_decision(
        endpoint="dispute", idempotency_key=idempotency_key,
        correlation_id=request.headers.get("x-request-id"),
        order_id=_order_id(body), user_id="dispute",
        image_base64=primary_image, image_phash=image_phash, response=resp,
        latency_ms=latency_ms,
        raw_observations={k: v for k, v in (observations or {}).items() if not k.startswith("_")},
        computed={"decision": resp.decision, "route": resp.route,
                  "refund": resp.refund.model_dump(), "agent_flags": resp.agent_flags},
    )
