import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.middleware.auth import verify_api_key
from app.middleware.rate_limit import limiter
from app.models.verify_request import VerifyClaimRequest
from app.models.verify_response import VerifyClaimResponse
from app.services.ai_image_detector import detect_ai_generated
from app.services.audit_service import build_idempotency_key, find_replay, persist_decision
from app.services.authenticity_engine import build_verify_response
from app.services.claim_service import analyze_claim
from app.services.dedup_service import find_duplicates, register_image
from app.utils.image_utils import compute_image_phash, validate_image_size
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

try:  # google-genai SDK error types (present in prod; guarded for safety)
    from google.genai import errors as genai_errors

    _UPSTREAM_API_ERROR = (genai_errors.APIError,)
except Exception:  # noqa: BLE001
    _UPSTREAM_API_ERROR = ()


@router.post(
    "/api/v1/imgrecog/verify-claim",
    response_model=VerifyClaimResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("100/minute")
async def verify_claim(request: Request, body: VerifyClaimRequest) -> VerifyClaimResponse:
    start = time.time()

    # 413 — payload too large
    try:
        validate_image_size(body.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    # Idempotency — a repeat claim returns the prior verdict verbatim.
    image_phash = compute_image_phash(body.image_base64)
    idempotency_key = build_idempotency_key(
        body.idempotency_key, body.claim_id, body.order_id, body.user_id,
        image_phash, body.image_base64,
    )
    replay = await find_replay("verify_claim", idempotency_key)
    if replay is not None:
        logger.info(
            "idempotent_replay",
            endpoint="verify_claim",
            order_id=body.order_id,
            request_id=replay.request_id,
        )
        return replay

    # Gemini observations — map upstream conditions to honest status codes.
    try:
        gemini = await analyze_claim(
            body.image_base64,
            body.user_comment,
            body.claimed_product,
            body.reference_image_base64,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Image analysis timed out")
    except HTTPException:
        raise
    except _UPSTREAM_API_ERROR as exc:  # type: ignore[misc]
        code = getattr(exc, "code", None)
        if code in (429, 503):
            logger.error("claim_quota_exhausted", code=code, order_id=body.order_id)
            raise HTTPException(
                status_code=503,
                detail="Claim analysis temporarily unavailable (upstream quota/overload)",
            )
        logger.error("claim_upstream_error", code=code, error=str(exc), order_id=body.order_id)
        raise HTTPException(status_code=502, detail="Upstream claim analysis error")
    except Exception as exc:  # noqa: BLE001
        logger.error("verify_failed", error=str(exc), order_id=body.order_id)
        raise HTTPException(status_code=500, detail="Claim analysis failed")

    # AI-generated check (free internal ensemble, or Sightengine if configured).
    ai_check = await detect_ai_generated(
        body.image_base64, gemini.get("ai_generated", {})
    )

    # Reused-image fraud check (hard signal) — see dedup_service.
    dedup_result = await find_duplicates(image_phash, body.order_id, body.user_id)
    response = build_verify_response(
        gemini, ai_check, body.order_id, body.user_id, dedup_result
    )

    latency_ms = round((time.time() - start) * 1000)
    response = await persist_decision(
        endpoint="verify_claim",
        idempotency_key=idempotency_key,
        correlation_id=request.headers.get("x-request-id"),
        order_id=body.order_id,
        user_id=body.user_id,
        image_base64=body.image_base64,
        image_phash=image_phash,
        response=response,
        latency_ms=latency_ms,
        raw_observations=gemini,
        computed={
            "authenticity_score": response.authenticity_score,
            "decision_confidence": response.decision_confidence,
            "dedup": dedup_result.to_audit(),
        },
    )

    # Record this claim's image so future claims can match it (best-effort).
    await register_image(image_phash, body.order_id, body.user_id, response.request_id)

    logger.info(
        "verify_complete",
        request_id=response.request_id,
        order_id=body.order_id,
        verdict=response.verdict,
        action=response.recommended_action,
        score=response.authenticity_score,
        ai_flagged=ai_check.is_ai_generated,
        latency_ms=latency_ms,
    )
    return response
