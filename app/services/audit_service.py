"""Idempotency + durable audit for every money-affecting decision.

Flow (both endpoints):
  1. build_idempotency_key(...)        — explicit key/claim_id, else derived from
                                         (order_id, user_id, image phash).
  2. find_replay(endpoint, key)        — HIT -> return the prior decision verbatim
                                         (no model call, no second action).
  3. persist_decision(...)             — upload image to object storage, write the
                                         audit row. On a concurrent duplicate, return
                                         the winner. On any audit-write failure,
                                         DOWNGRADE to manual review rather than
                                         auto-approving without a durable record.

Principle: a model output must never directly trigger a payout, and no decision is
returned to Kaily that isn't (or couldn't be) recorded.
"""

from __future__ import annotations

import base64
import hashlib
import uuid
from typing import Optional, Tuple

from pydantic import BaseModel

from app.config.settings import settings
from app.db.repository import (
    DecisionRecord,
    DuplicateDecision,
    get_decision_repository,
)
from app.models.response import ActionResult, ScanResponse
from app.models.verify_response import VerifyClaimResponse
from app.storage.object_store import get_object_store
from app.utils.image_utils import extract_base64_data
from app.utils.logger import get_logger

logger = get_logger(__name__)

_RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "scan": ScanResponse,
    "verify_claim": VerifyClaimResponse,
}


def build_idempotency_key(
    explicit_key: Optional[str],
    claim_id: Optional[str],
    order_id: str,
    user_id: str,
    image_phash: Optional[str],
    image_base64: str,
) -> str:
    """Caller-supplied key wins; else (order_id, claim_id); else a deterministic
    hash of (order_id, user_id, image identity). A perceptual hash is used as the
    image identity when available, otherwise an exact hash of the raw bytes."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    if claim_id and claim_id.strip():
        digest = hashlib.sha256(f"{order_id}|{claim_id.strip()}".encode()).hexdigest()
        return f"claim:{digest}"
    image_identity = image_phash or "raw:" + hashlib.sha256(image_base64.encode()).hexdigest()
    basis = f"{order_id}|{user_id}|{image_identity}"
    return "derived:" + hashlib.sha256(basis.encode()).hexdigest()


def _reconstruct(endpoint: str, snapshot: dict) -> BaseModel:
    return _RESPONSE_MODELS[endpoint].model_validate(snapshot)


async def find_replay(endpoint: str, idempotency_key: str) -> Optional[BaseModel]:
    """Return the previously-persisted decision for this key, or None."""
    repo = get_decision_repository()
    record = await repo.get_by_idempotency_key(idempotency_key)
    if record is None:
        return None
    return _reconstruct(endpoint, record.response_snapshot)


def _derive_routing(endpoint: str, response: BaseModel) -> Tuple[str, Optional[str], Optional[str], str]:
    """(final_action, final_status, priority, routed_to) flattened for querying/alerting."""
    if endpoint == "scan":
        assert isinstance(response, ScanResponse)
        routed_to = "human" if "manual review" in (response.action.message or "").lower() else "auto"
        return response.action.type, response.status, response.action.priority, routed_to
    assert isinstance(response, VerifyClaimResponse)
    routed_to = "human" if response.recommended_action == "manual_review" else "auto"
    return response.recommended_action, response.verdict, None, routed_to


def _downgrade(endpoint: str, response: BaseModel) -> BaseModel:
    """Safe fallback when the audit write fails: never auto-act, force human review."""
    if endpoint == "scan":
        return response.model_copy(
            update={
                "action": ActionResult(
                    type="no_action",
                    message=(
                        "Audit record could not be persisted; routing this claim to "
                        "manual review. No automated refund or exchange was initiated."
                    ),
                    refund_eligible=False,
                    priority="high",
                )
            }
        )
    note = " [Audit write failed — forced manual review; no automated action taken.]"
    return response.model_copy(
        update={
            "recommended_action": "manual_review",
            "verdict": "review",
            "agent_comment": (getattr(response, "agent_comment", "") or "") + note,
        }
    )


def _prompt_version(endpoint: str) -> str:
    return settings.scan_prompt_version if endpoint == "scan" else settings.verify_prompt_version


async def persist_decision(
    *,
    endpoint: str,
    idempotency_key: str,
    correlation_id: Optional[str],
    order_id: str,
    user_id: str,
    image_base64: str,
    image_phash: Optional[str],
    response: BaseModel,
    latency_ms: Optional[int],
    raw_observations: dict,
    computed: dict,
) -> BaseModel:
    """Persist the audit row (image -> object storage, metadata -> Postgres).

    Returns the response to send to Kaily: the original on success, the prior
    decision on a concurrent duplicate, or a manual-review downgrade if the audit
    write fails (a money mover with no record is worse than a slow one).
    """
    repo = get_decision_repository()
    store = get_object_store()
    try:
        raw_b64, mime_type = extract_base64_data(image_base64)
        image_bytes = base64.b64decode(raw_b64 + "==", validate=False)
        image_ref = await store.put(image_bytes, mime_type, prefix=endpoint)

        final_action, final_status, priority, routed_to = _derive_routing(endpoint, response)
        record = DecisionRecord(
            id=str(uuid.uuid4()),
            request_id=getattr(response, "request_id"),
            idempotency_key=idempotency_key,
            endpoint=endpoint,
            order_id=order_id,
            user_id=user_id,
            correlation_id=correlation_id,
            image_ref=image_ref,
            image_phash=image_phash,
            model_name=settings.gemini_model,
            model_version=settings.gemini_model,  # Phase 6 pins the dated version
            prompt_version=_prompt_version(endpoint),
            raw_observations=raw_observations or {},
            computed=computed or {},
            response_snapshot=response.model_dump(mode="json"),
            final_action=final_action,
            final_status=final_status,
            priority=priority,
            routed_to=routed_to,
            latency_ms=latency_ms,
        )
        await repo.insert(record)
        return response
    except DuplicateDecision:
        existing = await repo.get_by_idempotency_key(idempotency_key)
        if existing is not None:
            logger.info(
                "idempotent_replay_on_insert",
                endpoint=endpoint,
                order_id=order_id,
                idempotency_key=idempotency_key,
            )
            return _reconstruct(endpoint, existing.response_snapshot)
        return response
    except Exception as exc:  # noqa: BLE001 - any audit failure -> safe downgrade
        logger.error(
            "audit_write_failed",
            endpoint=endpoint,
            order_id=order_id,
            error=str(exc),
        )
        return _downgrade(endpoint, response)
