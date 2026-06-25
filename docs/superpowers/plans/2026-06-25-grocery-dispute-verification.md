# Grocery Dispute Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single `POST /api/v1/imgrecog/dispute` endpoint that resolves all 7 grocery dispute categories deterministically from customer images + ticket text + shipment data, returning approve/reject/agent with a refund amount and routing flags.

**Architecture:** Extends the existing FastAPI `imgrecog-kaily` service. One Gemini call returns image/OCR *observations*; a new deterministic `dispute_engine` computes every money decision. A keyword `category_classifier` resolves the dispute category (with the §3 fallback chain) without a model call. Reuses the existing audit/idempotency/dedup/object-store/auth/rate-limit spine. The 3 existing endpoints are untouched.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, `google-genai` SDK, SQLAlchemy async (Postgres), Redis, pytest. Spec: `docs/superpowers/specs/2026-06-25-grocery-dispute-verification-design.md`.

## Global Constraints

- **Money decisions are deterministic.** The model only observes; `dispute_engine.py` decides. Never let a Gemini field directly set `decision`/`refund`.
- **Match existing patterns.** Mirror `app/routers/verify.py`, `app/services/claim_service.py`, `app/services/decision_engine.py`. Pydantic v2 (`Field`, `field_validator`, `model_validate`).
- **Test isolation.** Mock Gemini at the service boundary (`app.services.dispute_service.analyze_dispute`). `tests/conftest.py` already pins `KAILY_API_SECRET=test-secret` and resets in-memory stores.
- **No PII in logs.** Log lengths/flags/ids, never raw ticket text.
- **MRP rule:** overcharge ⇔ printed MRP on pack (final/lowest value) `<` invoice MRP (`shipment.mrp`). Equality → reject.
- **Refund ceiling:** `REFUND_AUTO_APPROVE_MAX` (default `500`). Approved refund `>=` ceiling → agent.
- **Near-expiry:** non-FNV approve when `days_until_expiry <= NON_FNV_NEAR_EXPIRY_DAYS` (default 45); dairy approve when `shelf_left_pct < DAIRY_MIN_SHELF_PCT` (default 30); FNV expiry routes to poor-quality logic.
- **Run tests with:** `venv/bin/pytest -q` (Gemini mocked, no key/network needed).
- **Commit style:** single `-m` line, no body, no Claude co-author trailer.

---

### Task 1: Settings — new dispute config vars

**Files:**
- Modify: `app/config/settings.py`
- Modify: `.env.example`
- Test: `tests/test_dispute_settings.py`

**Interfaces:**
- Produces: `settings.refund_auto_approve_max: float`, `settings.dispute_assist_mode: bool`, `settings.dispute_autonomous_categories: str`, `settings.dairy_min_shelf_pct: float`, `settings.non_fnv_near_expiry_days: int`, `settings.dispute_max_images: int`, `settings.gemini_max_concurrency: int`, `settings.dispute_prompt_version: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispute_settings.py
from app.config.settings import settings


def test_dispute_defaults():
    assert settings.refund_auto_approve_max == 500
    assert settings.dispute_assist_mode is False
    assert "mrp_abuse" in settings.dispute_autonomous_categories
    assert settings.dairy_min_shelf_pct == 30
    assert settings.non_fnv_near_expiry_days == 45
    assert settings.dispute_max_images == 5
    assert settings.dispute_prompt_version == "dispute-v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_dispute_settings.py -q`
Expected: FAIL (`AttributeError: ... refund_auto_approve_max`).

- [ ] **Step 3: Add the settings fields**

In `app/config/settings.py`, after the `pixelbin_*` block (before `model_config`), add:

```python
    # ---- Grocery dispute verification (/dispute) ---------------------------
    # Approved refund >= this (INR) routes to a human agent with the AI
    # recommendation attached, even when the category decision is "approve".
    refund_auto_approve_max: float = 500
    # When true, every dispute decision becomes recommend-only (route=agent) —
    # a shadow/assist period without a redeploy.
    dispute_assist_mode: bool = False
    # Comma-separated categories allowed to auto-act. Others are recommend-only
    # until they clear the accuracy bar (progressive rollout).
    dispute_autonomous_categories: str = "mrp_abuse,expiry,wrong_product,damaged"
    # Dairy: approve near-expiry when remaining shelf life is below this percent.
    dairy_min_shelf_pct: float = 30
    # Non-FNV: approve when days until expiry is at or below this (Legal Metrology).
    non_fnv_near_expiry_days: int = 45
    # Max customer images accepted per dispute (bounded input).
    dispute_max_images: int = 5
    # Per-instance concurrency cap on Gemini calls (quota/backpressure).
    gemini_max_concurrency: int = 8
    dispute_prompt_version: str = "dispute-v1"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_dispute_settings.py -q`
Expected: PASS.

- [ ] **Step 5: Document the vars in `.env.example`**

Append to `.env.example`:

```bash
# ---- Grocery dispute verification (/dispute) ----
REFUND_AUTO_APPROVE_MAX=500
DISPUTE_ASSIST_MODE=false
DISPUTE_AUTONOMOUS_CATEGORIES=mrp_abuse,expiry,wrong_product,damaged
DAIRY_MIN_SHELF_PCT=30
NON_FNV_NEAR_EXPIRY_DAYS=45
DISPUTE_MAX_IMAGES=5
GEMINI_MAX_CONCURRENCY=8
DISPUTE_PROMPT_VERSION=dispute-v1
```

- [ ] **Step 6: Commit**

```bash
git add app/config/settings.py .env.example tests/test_dispute_settings.py
git commit -m "feat(dispute): add dispute verification config settings"
```

---

### Task 2: Request/response models

**Files:**
- Create: `app/models/dispute_request.py`
- Create: `app/models/dispute_response.py`
- Test: `tests/test_dispute_models.py`

**Interfaces:**
- Produces: `DisputeRequest{images: list[str], dispute_category: str|None, is_rebuttal: bool, ticket: Ticket, shipment: Shipment, idempotency_key, claim_id}`; `Ticket{title, description, notes, disposition_code}`; `Shipment{order_tracking_id, product_name, product_type, mrp, selling_price, invoice_amount, quantity, seller_type}`; `DisputeResponse{success, request_id, order_tracking_id, category, category_source, decision, route, agent_flags, refund: RefundResult, recommendation, confidence, observations, processed_at, model_used}`; `RefundResult{eligible, amount, type, assign_to_mpt, seller_debit}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispute_models.py
import pytest
from pydantic import ValidationError

from app.models.dispute_request import DisputeRequest
from app.models.dispute_response import DisputeResponse, RefundResult


def _req(**over):
    base = dict(
        images=["data:image/jpeg;base64,AAAA"],
        ticket={"title": "t", "description": "wrong item", "notes": "", "disposition_code": ""},
        shipment={
            "order_tracking_id": "JM-1", "product_name": "Amul Milk",
            "product_type": "dairy", "mrp": 33.0, "selling_price": 31.0,
            "invoice_amount": 62.0, "quantity": 2, "seller_type": "1P",
        },
    )
    base.update(over)
    return DisputeRequest(**base)


def test_request_minimal_ok():
    r = _req()
    assert r.shipment.seller_type == "1P"
    assert r.is_rebuttal is False
    assert r.dispute_category is None


def test_request_rejects_empty_images():
    with pytest.raises(ValidationError):
        _req(images=[])


def test_request_rejects_bad_product_type():
    with pytest.raises(ValidationError):
        _req(shipment={**_req().shipment.model_dump(), "product_type": "gadget"})


def test_response_defaults():
    resp = DisputeResponse(
        request_id="dsp_1", order_tracking_id="JM-1", category="mrp_abuse",
        category_source="provided", decision="approve", route="auto",
        refund=RefundResult(eligible=True, amount=4.0, type="price_difference"),
        recommendation="ok", confidence=0.9, observations={}, model_used="gemini-2.5-flash",
    )
    assert resp.success is True
    assert resp.agent_flags == []
    assert resp.refund.assign_to_mpt is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_dispute_models.py -q`
Expected: FAIL (`ModuleNotFoundError: app.models.dispute_request`).

- [ ] **Step 3: Create `app/models/dispute_request.py`**

```python
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ProductType = Literal["fnv", "non_fnv", "dairy"]
SellerType = Literal["1P", "3P"]
DisputeCategory = Literal[
    "wrong_product", "poor_quality", "damaged", "expiry",
    "smell", "mrp_abuse", "quantity_mismatch",
]


class Ticket(BaseModel):
    title: str = Field(default="", max_length=500)
    description: str = Field(default="", max_length=4000)
    notes: str = Field(default="", max_length=4000)
    disposition_code: str = Field(default="", max_length=100)


class Shipment(BaseModel):
    order_tracking_id: str = Field(..., min_length=1)
    product_name: str = Field(..., min_length=1, max_length=500)
    product_type: ProductType
    mrp: Optional[float] = Field(default=None, ge=0)
    selling_price: Optional[float] = Field(default=None, ge=0)
    invoice_amount: Optional[float] = Field(default=None, ge=0)
    quantity: int = Field(default=1, ge=1)
    seller_type: SellerType = "1P"


class DisputeRequest(BaseModel):
    """A grocery dispute to resolve from images + ticket text + shipment data."""

    images: List[str] = Field(..., min_length=1)
    dispute_category: Optional[DisputeCategory] = None
    is_rebuttal: bool = False
    ticket: Ticket = Field(default_factory=Ticket)
    shipment: Shipment
    idempotency_key: Optional[str] = Field(default=None, max_length=200)
    claim_id: Optional[str] = Field(default=None, max_length=200)

    @field_validator("images")
    @classmethod
    def _non_empty_images(cls, v: List[str]) -> List[str]:
        cleaned = [s for s in v if s and s.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty image is required")
        return cleaned
```

- [ ] **Step 4: Create `app/models/dispute_response.py`**

```python
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Decision = Literal["approve", "reject", "agent"]
Route = Literal["auto", "agent"]
RefundType = Literal["price_difference", "full_selling_price", "none"]
CategorySource = Literal["provided", "description", "notes", "disposition", "none"]


class RefundResult(BaseModel):
    eligible: bool = False
    amount: float = 0.0
    type: RefundType = "none"
    assign_to_mpt: bool = False
    seller_debit: bool = False


class DisputeResponse(BaseModel):
    success: bool = True
    request_id: str
    order_tracking_id: str
    category: Optional[str] = None
    category_source: CategorySource = "none"
    decision: Decision
    route: Route
    agent_flags: List[str] = Field(default_factory=list)
    refund: RefundResult = Field(default_factory=RefundResult)
    recommendation: str = ""
    confidence: float = 0.0
    observations: Dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_used: str = ""
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_dispute_models.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/models/dispute_request.py app/models/dispute_response.py tests/test_dispute_models.py
git commit -m "feat(dispute): add request/response models"
```

---

### Task 3: Category classifier (fallback chain §3)

**Files:**
- Create: `app/services/category_classifier.py`
- Test: `tests/test_category_classifier.py`

**Interfaces:**
- Consumes: `Ticket` (Task 2), `DisputeCategory`.
- Produces: `classify_category(dispute_category: Optional[str], ticket: Ticket) -> tuple[Optional[str], str]` returning `(category, source)` where source ∈ `{provided, description, notes, disposition, none}`. `category is None` ⇒ INSUFFICIENT_DATA.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_category_classifier.py
from app.models.dispute_request import Ticket
from app.services.category_classifier import classify_category


def test_provided_wins():
    assert classify_category("mrp_abuse", Ticket(description="anything")) == ("mrp_abuse", "provided")


def test_description_keyword():
    assert classify_category(None, Ticket(description="I got the wrong item")) == ("wrong_product", "description")


def test_notes_when_description_blank():
    t = Ticket(description="", notes="bottle was leaking everywhere")
    assert classify_category(None, t) == ("damaged", "notes")


def test_disposition_map():
    t = Ticket(description="", notes="", disposition_code="PRICE_DISPUTE")
    assert classify_category(None, t) == ("mrp_abuse", "disposition")


def test_insufficient_data():
    assert classify_category(None, Ticket()) == (None, "none")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_category_classifier.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Create `app/services/category_classifier.py`**

```python
"""Resolve the dispute category via the requirement's §3 fallback chain.

Deterministic and model-free: explicit category -> keyword match on the
description -> keyword match on the notes -> disposition-code map -> None
(INSUFFICIENT_DATA). Keeping this in plain Python keeps it fast and testable.
"""

from typing import Optional, Tuple

from app.models.dispute_request import Ticket

# Order matters: earlier categories win when multiple keyword groups match.
_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("mrp_abuse", ("mrp", "overcharge", "over charge", "overcharged", "price", "charged more", "expensive")),
    ("expiry", ("expir", "expiry", "expired", "best before", "use by", "near expiry", "date over")),
    ("damaged", ("damage", "broken", "leak", "torn", "tear", "crush", "tamper", "seal", "spill", "dent")),
    ("smell", ("smell", "stink", "odor", "odour", "foul", "rotten")),
    ("wrong_product", ("wrong", "incorrect", "different item", "not what i ordered", "mismatch item")),
    ("quantity_mismatch", ("quantity", "missing item", "less item", "fewer", "short", "only got", "one less")),
    ("poor_quality", ("quality", "stale", "spoil", "bad", "wilt", "discolor", "fungus", "mold", "fresh")),
]

_DISPOSITION_MAP = {
    "WRONG_ITEM": "wrong_product",
    "QUALITY_ISSUE": "poor_quality",
    "DAMAGE": "damaged",
    "EXPIRY": "expiry",
    "PRICE_DISPUTE": "mrp_abuse",
}


def _match_keywords(text: str) -> Optional[str]:
    low = (text or "").lower()
    if not low.strip():
        return None
    for category, words in _KEYWORDS:
        if any(w in low for w in words):
            return category
    return None


def classify_category(
    dispute_category: Optional[str], ticket: Ticket
) -> Tuple[Optional[str], str]:
    if dispute_category:
        return dispute_category, "provided"
    by_desc = _match_keywords(ticket.description)
    if by_desc:
        return by_desc, "description"
    by_notes = _match_keywords(ticket.notes)
    if by_notes:
        return by_notes, "notes"
    mapped = _DISPOSITION_MAP.get((ticket.disposition_code or "").strip().upper())
    if mapped:
        return mapped, "disposition"
    return None, "none"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_category_classifier.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/category_classifier.py tests/test_category_classifier.py
git commit -m "feat(dispute): add fallback-chain category classifier"
```

---

### Task 4: OCR parser extensions (MRP + dairy shelf %)

**Files:**
- Modify: `app/services/ocr_parser.py`
- Test: `tests/test_ocr_parser_dispute.py`

**Interfaces:**
- Produces: `final_printed_mrp(values: list) -> Optional[float]` (lowest positive = post-strikethrough final price); `days_until_expiry(expiry_iso, today=None) -> Optional[int]` (negative if already expired); `shelf_left_pct(mfg_iso, exp_iso, today=None) -> Optional[float]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ocr_parser_dispute.py
from datetime import date

from app.services.ocr_parser import days_until_expiry, final_printed_mrp, shelf_left_pct


def test_final_printed_mrp_picks_lowest():
    assert final_printed_mrp([49.0, 31.0]) == 31.0


def test_final_printed_mrp_ignores_nonpositive_and_empty():
    assert final_printed_mrp([0, -5]) is None
    assert final_printed_mrp([]) is None


def test_days_until_expiry_future_and_past():
    today = date(2026, 6, 25)
    assert days_until_expiry("2026-07-10", today) == 15
    assert days_until_expiry("2026-06-20", today) == -5
    assert days_until_expiry(None, today) is None


def test_shelf_left_pct():
    today = date(2026, 6, 25)
    # mfg 2026-06-20, exp 2026-06-30 -> total 10d, 5 left -> 50%
    assert shelf_left_pct("2026-06-20", "2026-06-30", today) == 50.0
    assert shelf_left_pct(None, "2026-06-30", today) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_ocr_parser_dispute.py -q`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Append to `app/services/ocr_parser.py`**

```python
def final_printed_mrp(values: object) -> Optional[float]:
    """Pick the final (post-strikethrough) MRP from OCR'd candidate values.

    A reduced-price pack prints both the old (struck-through, higher) and the
    new (lower) MRP. The lowest positive value is the one the customer pays
    against, so it is the right basis for an overcharge comparison.
    """
    if not isinstance(values, (list, tuple)):
        return None
    nums: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            nums.append(f)
    return min(nums) if nums else None


def days_until_expiry(expiry_iso: Optional[str], today: Optional[date] = None) -> Optional[int]:
    """Whole days from today until expiry. Negative if already expired, None if unusable."""
    if not expiry_iso:
        return None
    try:
        expiry = date.fromisoformat(expiry_iso)
    except ValueError:
        return None
    return (expiry - (today or date.today())).days


def shelf_left_pct(
    mfg_iso: Optional[str], exp_iso: Optional[str], today: Optional[date] = None
) -> Optional[float]:
    """Remaining shelf life as a percent of total shelf life. None if unusable."""
    if not mfg_iso or not exp_iso:
        return None
    try:
        mfg = date.fromisoformat(mfg_iso)
        exp = date.fromisoformat(exp_iso)
    except ValueError:
        return None
    total = (exp - mfg).days
    if total <= 0:
        return None
    remaining = (exp - (today or date.today())).days
    return round(remaining / total * 100, 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_ocr_parser_dispute.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/ocr_parser.py tests/test_ocr_parser_dispute.py
git commit -m "feat(dispute): add MRP and dairy shelf-life OCR helpers"
```

---

### Task 5: Damage taxonomy extension

**Files:**
- Modify: `app/services/damage_analyzer.py:10-18`
- Test: `tests/test_damage_taxonomy_dispute.py`

**Interfaces:**
- Produces: `VALID_TYPES` now includes `tamper`, `resealed`, `missing_component` (in addition to existing `crushed_packaging, tear, broken_seal, leakage, dent, discoloration, mold`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_damage_taxonomy_dispute.py
from app.services.damage_analyzer import VALID_TYPES, normalize_damage


def test_new_tamper_types_accepted():
    for t in ("tamper", "resealed", "missing_component"):
        assert t in VALID_TYPES
        out = normalize_damage({"detected": True, "type": t, "severity": "severe"})
        assert out["type"] == t
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_damage_taxonomy_dispute.py -q`
Expected: FAIL (`tamper not in VALID_TYPES`).

- [ ] **Step 3: Extend `VALID_TYPES` in `app/services/damage_analyzer.py`**

Replace the `VALID_TYPES` set (lines 10-18) with:

```python
VALID_TYPES = {
    "crushed_packaging",
    "tear",
    "broken_seal",
    "leakage",
    "dent",
    "discoloration",
    "mold",
    "tamper",
    "resealed",
    "missing_component",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_damage_taxonomy_dispute.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/damage_analyzer.py tests/test_damage_taxonomy_dispute.py
git commit -m "feat(dispute): extend damage taxonomy with tamper types"
```

---

### Task 6: Dispute engine — MRP abuse + refund math

**Files:**
- Create: `app/services/dispute_engine.py`
- Test: `tests/test_dispute_engine_mrp.py`

**Interfaces:**
- Consumes: `final_printed_mrp` (Task 4); `Shipment` (Task 2).
- Produces: `DisputeDecision` dataclass `{decision: str, refund: dict, agent_flags: list[str], confidence: float, recommendation: str}` and `decide(category, source, observations, shipment, is_rebuttal, signals) -> DisputeDecision`. This task implements the `mrp_abuse` branch + refund math + the shared escalation gates; later tasks add other category branches. `signals` is a dict `{ai_probability: float, dedup_cross: bool, web_hard: bool}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispute_engine_mrp.py
from app.models.dispute_request import Shipment
from app.services.dispute_engine import decide


def _ship(**o):
    base = dict(order_tracking_id="JM-1", product_name="Oil 1L", product_type="non_fnv",
                mrp=100.0, selling_price=100.0, invoice_amount=200.0, quantity=2, seller_type="1P")
    base.update(o)
    return Shipment(**base)


_NO_SIGNALS = {"ai_probability": 0.0, "dedup_cross": False, "web_hard": False}


def test_mrp_overcharge_1p_refunds_difference():
    obs = {"ocr": {"printed_mrp_values": [90.0]}}  # printed 90 < invoice mrp 100 -> overcharge
    d = decide("mrp_abuse", "provided", obs, _ship(), False, _NO_SIGNALS)
    assert d.decision == "approve"
    assert d.refund["type"] == "price_difference"
    assert d.refund["amount"] == 20.0  # (100 charged - 90 printed) * 2
    assert d.refund["assign_to_mpt"] is False


def test_mrp_no_overcharge_rejects():
    obs = {"ocr": {"printed_mrp_values": [100.0]}}  # printed == invoice mrp -> reject
    d = decide("mrp_abuse", "provided", obs, _ship(), False, _NO_SIGNALS)
    assert d.decision == "reject"
    assert d.refund["eligible"] is False


def test_mrp_3p_full_refund_and_mpt():
    obs = {"ocr": {"printed_mrp_values": [90.0]}}
    d = decide("mrp_abuse", "provided", obs, _ship(seller_type="3P"), False, _NO_SIGNALS)
    assert d.decision == "agent"  # 100*2 = 200 >= 500? no -> stays approve unless ceiling; here 200<500
    # full refund computed regardless of routing
    assert d.refund["type"] == "full_selling_price"
    assert d.refund["amount"] == 200.0
    assert d.refund["assign_to_mpt"] is True
    assert d.refund["seller_debit"] is True


def test_mrp_unreadable_routes_agent():
    obs = {"ocr": {"printed_mrp_values": []}}
    d = decide("mrp_abuse", "provided", obs, _ship(), False, _NO_SIGNALS)
    assert d.decision == "agent"
    assert "missing_shipment_data" in d.agent_flags or "low_confidence" in d.agent_flags
```

> Note: `test_mrp_3p_full_refund_and_mpt` asserts `agent` because 3P approved overcharge below the ceiling still auto-acts — adjust the expectation after reading the escalation logic in Step 3 (3P with refund 200 < 500 and `mrp_abuse` autonomous ⇒ `approve`). Replace `assert d.decision == "agent"` with `assert d.decision == "approve"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_dispute_engine_mrp.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Create `app/services/dispute_engine.py`**

```python
"""Deterministic dispute decisions. The model observes; this engine decides.

decide() dispatches per category to a pure function that returns a base
DisputeDecision (approve/reject + refund), then applies the shared escalation
gates (counterfeit, rebuttal, fraud signals, refund ceiling, assist/autonomous).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.config.settings import settings
from app.models.dispute_request import Shipment
from app.services.ocr_parser import final_printed_mrp


@dataclass
class DisputeDecision:
    decision: str  # approve | reject | agent
    refund: dict
    agent_flags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    recommendation: str = ""


def _no_refund() -> dict:
    return {"eligible": False, "amount": 0.0, "type": "none",
            "assign_to_mpt": False, "seller_debit": False}


def _full_refund(shipment: Shipment) -> dict:
    unit = shipment.selling_price if shipment.selling_price is not None else 0.0
    amount = round(unit * shipment.quantity, 2)
    refund = {"eligible": True, "amount": amount, "type": "full_selling_price",
              "assign_to_mpt": False, "seller_debit": False}
    if shipment.seller_type == "3P":
        refund["assign_to_mpt"] = True
        refund["seller_debit"] = True
    return refund


def _agent(flag: str, recommendation: str, refund: Optional[dict] = None,
           confidence: float = 0.4) -> DisputeDecision:
    return DisputeDecision(decision="agent", refund=refund or _no_refund(),
                           agent_flags=[flag], confidence=confidence,
                           recommendation=recommendation)


# ---- per-category branches -------------------------------------------------

def _decide_mrp(observations: dict, shipment: Shipment) -> DisputeDecision:
    values = (observations.get("ocr") or {}).get("printed_mrp_values")
    printed = final_printed_mrp(values)
    if printed is None:
        return _agent("low_confidence", "MRP not readable from the image; agent to verify.")
    if shipment.mrp is None:
        return _agent("missing_shipment_data", "Invoice MRP missing; agent to verify overcharge.")
    if printed >= shipment.mrp:
        return DisputeDecision(
            decision="reject", refund=_no_refund(), confidence=0.9,
            recommendation=(f"Reject: printed MRP ₹{printed} ≥ invoice MRP ₹{shipment.mrp}; "
                            "no overcharge."),
        )
    # Overcharge confirmed: printed MRP on pack is below the MRP charged.
    if shipment.seller_type == "3P":
        refund = _full_refund(shipment)
        rec = (f"Approve full refund ₹{refund['amount']} (3P overcharge: printed ₹{printed} < "
               f"invoice MRP ₹{shipment.mrp}); assign ticket to MPT to debit seller.")
        return DisputeDecision(decision="approve", refund=refund, confidence=0.9, recommendation=rec)
    charged_unit = shipment.selling_price if shipment.selling_price is not None else shipment.mrp
    extra = round(max(0.0, charged_unit - printed) * shipment.quantity, 2)
    refund = {"eligible": extra > 0, "amount": extra, "type": "price_difference",
              "assign_to_mpt": False, "seller_debit": False}
    rec = (f"Approve ₹{extra} price-difference refund (1P overcharge: printed ₹{printed} < "
           f"invoice MRP ₹{shipment.mrp} × qty {shipment.quantity}).")
    return DisputeDecision(decision="approve", refund=refund, confidence=0.9, recommendation=rec)


_BRANCHES = {
    "mrp_abuse": _decide_mrp,
}


# ---- escalation gates ------------------------------------------------------

def _autonomous_categories() -> set:
    return {c.strip() for c in settings.dispute_autonomous_categories.split(",") if c.strip()}


def _apply_gates(base: DisputeDecision, category: str, shipment: Shipment,
                 observations: dict, is_rebuttal: bool, signals: dict) -> DisputeDecision:
    flags = list(base.agent_flags)
    decision = base.decision

    # Hard fraud signals -> reject (defensible, auditable).
    if signals.get("dedup_cross") or signals.get("web_hard"):
        return DisputeDecision(decision="reject", refund=_no_refund(),
                               agent_flags=["fraud_signal"], confidence=1.0,
                               recommendation="Reject: image reused across claims / found on public web.")

    # Counterfeit & rebuttal & advisory-AI -> always human.
    if observations.get("counterfeit_suspected"):
        return _agent("counterfeit", "Counterfeit suspected; route to agent investigation.",
                      refund=base.refund)
    if is_rebuttal:
        return _agent("rebuttal", f"Post-rejection rebuttal. AI recommendation: {base.recommendation}",
                      refund=base.refund)
    if signals.get("ai_probability", 0.0) >= settings.ai_detection_min_confidence:
        return _agent("fraud_signal", "Image may be AI-generated; agent to verify. "
                      f"AI recommendation: {base.recommendation}", refund=base.refund)

    if decision == "agent":
        return DisputeDecision(decision="agent", refund=base.refund, agent_flags=flags or ["low_confidence"],
                               confidence=base.confidence, recommendation=base.recommendation)

    # Refund ceiling.
    if decision == "approve" and base.refund.get("amount", 0.0) >= settings.refund_auto_approve_max:
        flags.append("high_value")
        return DisputeDecision(decision="agent", refund=base.refund, agent_flags=flags,
                               confidence=base.confidence,
                               recommendation=f"High-value refund. AI recommendation: {base.recommendation}")

    # Assist mode / progressive autonomy -> recommend-only.
    if settings.dispute_assist_mode or category not in _autonomous_categories():
        return DisputeDecision(decision=decision, refund=base.refund,
                               agent_flags=flags, confidence=base.confidence,
                               recommendation=base.recommendation)
    return DisputeDecision(decision=decision, refund=base.refund, agent_flags=flags,
                           confidence=base.confidence, recommendation=base.recommendation)


def decide(category: Optional[str], source: str, observations: dict, shipment: Shipment,
           is_rebuttal: bool, signals: dict) -> DisputeDecision:
    if not category:
        return _agent("insufficient_data",
                      "No category could be resolved from description, notes, or disposition.")
    branch = _BRANCHES.get(category)
    if branch is None:
        return _agent("low_confidence", f"Category '{category}' not yet automated; route to agent.")
    base = branch(observations, shipment)
    decided = _apply_gates(base, category, shipment, observations, is_rebuttal, signals)
    # When assist/non-autonomous, the route (set in the router) becomes agent but
    # the decision stays as the AI's recommendation. The router maps that to route.
    return decided
```

> Note for the implementer: in `_apply_gates`, the assist-mode / non-autonomous branch keeps `decision` as the recommendation (approve/reject) — the **router** sets `route="agent"` based on `decision == "agent"` OR assist/non-autonomous. The engine returns the recommendation; routing is computed in Task 11/12. For now the test only checks `decision`, and `mrp_abuse` is in the autonomous list, so approve/reject stand.

- [ ] **Step 4: Fix the 3P test expectation**

In `tests/test_dispute_engine_mrp.py::test_mrp_3p_full_refund_and_mpt`, change
`assert d.decision == "agent"` to `assert d.decision == "approve"`.

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_dispute_engine_mrp.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/dispute_engine.py tests/test_dispute_engine_mrp.py
git commit -m "feat(dispute): dispute engine with MRP abuse logic and escalation gates"
```

---

### Task 7: Dispute engine — expiry (non-FNV 45d, dairy 30%, FNV→quality)

**Files:**
- Modify: `app/services/dispute_engine.py` (add `_decide_expiry`, register in `_BRANCHES`; import helpers)
- Test: `tests/test_dispute_engine_expiry.py`

**Interfaces:**
- Consumes: `days_until_expiry`, `shelf_left_pct` (Task 4).
- Produces: `_decide_expiry(observations, shipment)` registered for `"expiry"`. Reads `observations["ocr"]["expiry_date"]`, `["manufacture_date"]`. For `product_type == "fnv"`, delegates to the poor-quality branch (added in Task 8) via a module-level dispatch; until Task 8 exists, FNV expiry returns agent `low_confidence`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispute_engine_expiry.py
from app.models.dispute_request import Shipment
from app.services.dispute_engine import decide

_NO = {"ai_probability": 0.0, "dedup_cross": False, "web_hard": False}


def _ship(pt="non_fnv", **o):
    base = dict(order_tracking_id="JM-1", product_name="Biscuits", product_type=pt,
                mrp=50.0, selling_price=50.0, invoice_amount=50.0, quantity=1, seller_type="1P")
    base.update(o)
    return Shipment(**base)


def test_non_fnv_near_expiry_approves(monkeypatch):
    from app.services import dispute_engine
    obs = {"ocr": {"expiry_date": "2026-07-01"}}  # ~6 days from 2026-06-25
    monkeypatch.setattr(dispute_engine, "_today", lambda: __import__("datetime").date(2026, 6, 25))
    d = decide("expiry", "provided", obs, _ship(), False, _NO)
    assert d.decision == "approve"


def test_non_fnv_far_expiry_rejects(monkeypatch):
    from app.services import dispute_engine
    obs = {"ocr": {"expiry_date": "2026-12-31"}}
    monkeypatch.setattr(dispute_engine, "_today", lambda: __import__("datetime").date(2026, 6, 25))
    d = decide("expiry", "provided", obs, _ship(), False, _NO)
    assert d.decision == "reject"


def test_dairy_low_shelf_approves(monkeypatch):
    from app.services import dispute_engine
    obs = {"ocr": {"manufacture_date": "2026-06-20", "expiry_date": "2026-06-30"}}  # 10d total
    monkeypatch.setattr(dispute_engine, "_today", lambda: __import__("datetime").date(2026, 6, 28))
    d = decide("expiry", "provided", obs, _ship(pt="dairy"), False, _NO)  # 2/10 = 20% < 30
    assert d.decision == "approve"


def test_expiry_unreadable_agent():
    d = decide("expiry", "provided", {"ocr": {}}, _ship(), False, _NO)
    assert d.decision == "agent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_dispute_engine_expiry.py -q`
Expected: FAIL (`KeyError`/agent for all — `expiry` not in `_BRANCHES`).

- [ ] **Step 3: Add expiry logic to `app/services/dispute_engine.py`**

Add imports at the top (next to the existing `final_printed_mrp` import):

```python
from datetime import date

from app.services.ocr_parser import days_until_expiry, shelf_left_pct
```

Add a seam for "today" (so tests can pin it) and the branch:

```python
def _today() -> date:
    return date.today()


def _decide_expiry(observations: dict, shipment: Shipment) -> DisputeDecision:
    ocr = observations.get("ocr") or {}
    if shipment.product_type == "fnv":
        # FNV has variable shelf life — judge by visible quality instead.
        return _decide_poor_quality(observations, shipment)

    if shipment.product_type == "dairy":
        pct = shelf_left_pct(ocr.get("manufacture_date"), ocr.get("expiry_date"), _today())
        if pct is None:
            return _agent("low_confidence", "Dairy MFG/EXP not readable; agent to verify shelf life.")
        if pct < settings.dairy_min_shelf_pct:
            return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.9,
                                   recommendation=f"Approve: dairy shelf life {pct}% < {settings.dairy_min_shelf_pct}%.")
        return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.9,
                               recommendation=f"Reject: dairy shelf life {pct}% ≥ {settings.dairy_min_shelf_pct}%.")

    days = days_until_expiry(ocr.get("expiry_date"), _today())
    if days is None:
        return _agent("low_confidence", "Expiry date not readable; agent to verify.")
    if days <= settings.non_fnv_near_expiry_days:
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.9,
                               recommendation=f"Approve: {days} days to expiry ≤ {settings.non_fnv_near_expiry_days}.")
    return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.9,
                           recommendation=f"Reject: {days} days to expiry > {settings.non_fnv_near_expiry_days}.")
```

Register it: change `_BRANCHES` to include expiry (place AFTER `_decide_poor_quality` is defined in Task 8; for now reference it — define a temporary `_decide_poor_quality` stub at the end of this task so FNV works, then Task 8 replaces it):

```python
def _decide_poor_quality(observations: dict, shipment: Shipment) -> DisputeDecision:
    # Replaced with full logic in Task 8.
    return _agent("low_confidence", "Quality assessment pending.")


_BRANCHES = {
    "mrp_abuse": _decide_mrp,
    "expiry": _decide_expiry,
}
```

> Implementer note: move the `_BRANCHES` dict to the BOTTOM of the file (after all `_decide_*` are defined) to avoid forward-reference errors. Keep a single `_BRANCHES` definition.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_dispute_engine_expiry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/dispute_engine.py tests/test_dispute_engine_expiry.py
git commit -m "feat(dispute): expiry/dairy/FNV decision logic"
```

---

### Task 8: Dispute engine — wrong_product, damaged, poor_quality, smell, quantity_mismatch

**Files:**
- Modify: `app/services/dispute_engine.py` (replace `_decide_poor_quality` stub; add the others; finalize `_BRANCHES`)
- Test: `tests/test_dispute_engine_categories.py`

**Interfaces:**
- Consumes: `normalize_damage` (existing).
- Produces: `_decide_wrong_product`, `_decide_damaged`, `_decide_poor_quality`, `_decide_smell`, `_decide_quantity` registered in `_BRANCHES`. Observation contract: `product_match.matches: bool`; `damage{detected,type,severity}`; `quality{poor_quality, supports_claim, internal_defect}`; `spoilage.mold_or_visible_spoilage: bool`; `count{counted_units: int|None, confidence: float}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispute_engine_categories.py
from app.models.dispute_request import Shipment
from app.services.dispute_engine import decide

_NO = {"ai_probability": 0.0, "dedup_cross": False, "web_hard": False}


def _ship(pt="non_fnv", **o):
    base = dict(order_tracking_id="JM-1", product_name="Tata Salt 1kg", product_type=pt,
                mrp=28.0, selling_price=28.0, invoice_amount=28.0, quantity=3, seller_type="1P")
    base.update(o)
    return Shipment(**base)


def test_wrong_product_mismatch_approves():
    d = decide("wrong_product", "provided", {"product_match": {"matches": False}}, _ship(), False, _NO)
    assert d.decision == "approve"


def test_wrong_product_match_rejects():
    d = decide("wrong_product", "provided", {"product_match": {"matches": True}}, _ship(), False, _NO)
    assert d.decision == "reject"


def test_damaged_confirmed_approves():
    obs = {"damage": {"detected": True, "type": "leakage", "severity": "severe"}}
    d = decide("damaged", "provided", obs, _ship(), False, _NO)
    assert d.decision == "approve"


def test_damaged_intact_rejects():
    d = decide("damaged", "provided", {"damage": {"detected": False}}, _ship(), False, _NO)
    assert d.decision == "reject"


def test_poor_quality_internal_defect_to_agent():
    obs = {"quality": {"poor_quality": True, "supports_claim": True, "internal_defect": True}}
    d = decide("poor_quality", "provided", obs, _ship(), False, _NO)
    assert d.decision == "agent"
    assert "internal_defect" in d.agent_flags


def test_quantity_short_approves():
    obs = {"count": {"counted_units": 2, "confidence": 0.9}}  # ordered 3
    d = decide("quantity_mismatch", "provided", obs, _ship(), False, _NO)
    assert d.decision == "approve"


def test_quantity_low_confidence_agent():
    obs = {"count": {"counted_units": None, "confidence": 0.2}}
    d = decide("quantity_mismatch", "provided", obs, _ship(), False, _NO)
    assert d.decision == "agent"


def test_smell_with_spoilage_and_detail_approves():
    obs = {"spoilage": {"mold_or_visible_spoilage": True}, "_desc_len": 60}
    d = decide("smell", "provided", obs, _ship(), False, _NO)
    assert d.decision == "approve"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_dispute_engine_categories.py -q`
Expected: FAIL (branches missing / stub returns agent).

- [ ] **Step 3: Replace the stub and add branches in `app/services/dispute_engine.py`**

Add the import:

```python
from app.services.damage_analyzer import normalize_damage
```

Replace the temporary `_decide_poor_quality` and add the rest:

```python
def _decide_wrong_product(observations: dict, shipment: Shipment) -> DisputeDecision:
    pm = observations.get("product_match") or {}
    if pm.get("matches") is True:
        return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.85,
                               recommendation="Reject: delivered product matches the ordered item.")
    if pm.get("matches") is False:
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.85,
                               recommendation="Approve: delivered product does not match the ordered item.")
    return _agent("low_confidence", "Could not confirm product match; agent to verify.")


def _decide_damaged(observations: dict, shipment: Shipment) -> DisputeDecision:
    dmg = normalize_damage(observations.get("damage") or {})
    if dmg.get("detected"):
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.85,
                               recommendation=f"Approve: {dmg.get('type') or 'damage'} visually confirmed.")
    return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.8,
                           recommendation="Reject: packaging appears intact, no damage visible.")


def _decide_poor_quality(observations: dict, shipment: Shipment) -> DisputeDecision:
    q = observations.get("quality") or {}
    if q.get("internal_defect"):
        return _agent("internal_defect",
                      "Internal defect / warranty / performance issue; route to agent.")
    if q.get("poor_quality") and q.get("supports_claim"):
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.75,
                               recommendation="Approve: visible quality defect supports the claim.")
    return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.7,
                           recommendation="Reject: product appears normal in the image.")


def _decide_smell(observations: dict, shipment: Shipment) -> DisputeDecision:
    spoil = (observations.get("spoilage") or {}).get("mold_or_visible_spoilage")
    detailed = int(observations.get("_desc_len", 0)) >= 30
    if spoil and detailed:
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=0.65,
                               recommendation="Approve: detailed report plus visible spoilage proxy.")
    return DisputeDecision(decision="reject", refund=_no_refund(), confidence=0.6,
                           recommendation="Reject: insufficient evidence for a smell claim.")


def _decide_quantity(observations: dict, shipment: Shipment) -> DisputeDecision:
    count = observations.get("count") or {}
    units = count.get("counted_units")
    conf = float(count.get("confidence", 0.0))
    if units is None or conf < 0.6:
        return _agent("low_confidence", "Could not count units confidently; agent to verify quantity.")
    if int(units) < shipment.quantity:
        return DisputeDecision(decision="approve", refund=_full_refund(shipment), confidence=conf,
                               recommendation=f"Approve: counted {int(units)} < ordered {shipment.quantity}.")
    return DisputeDecision(decision="reject", refund=_no_refund(), confidence=conf,
                           recommendation=f"Reject: counted {int(units)} ≥ ordered {shipment.quantity}.")
```

Finalize the single `_BRANCHES` dict at the bottom of the file:

```python
_BRANCHES = {
    "mrp_abuse": _decide_mrp,
    "expiry": _decide_expiry,
    "wrong_product": _decide_wrong_product,
    "damaged": _decide_damaged,
    "poor_quality": _decide_poor_quality,
    "smell": _decide_smell,
    "quantity_mismatch": _decide_quantity,
}
```

- [ ] **Step 4: Run the full engine suite**

Run: `venv/bin/pytest tests/test_dispute_engine_categories.py tests/test_dispute_engine_mrp.py tests/test_dispute_engine_expiry.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add app/services/dispute_engine.py tests/test_dispute_engine_categories.py
git commit -m "feat(dispute): wrong-product/damaged/quality/smell/quantity logic"
```

---

### Task 9: Dispute service (Gemini multi-image observation call)

**Files:**
- Create: `app/services/dispute_service.py`
- Modify: `app/services/gemini_service.py` (add `build_dispute_generation_config`)
- Test: `tests/test_dispute_service.py`

**Interfaces:**
- Consumes: `generate_content_with_fallback`, `extract_base64_data`.
- Produces: `async analyze_dispute(images: list[str], category: Optional[str], product_name: str, comment: str) -> dict` returning the observation JSON described in the spec (`ocr.printed_mrp_values`, `ocr.expiry_date`, `ocr.manufacture_date`, `product_match.matches`, `damage{}`, `quality{}`, `spoilage{}`, `count{}`, `counterfeit_suspected`, `ai_generated.ai_probability`). Raises `TimeoutError` on timeout, `ValueError` on bad/empty/malformed image or JSON.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispute_service.py
import asyncio
import json
from unittest.mock import patch

import pytest

from app.services.dispute_service import analyze_dispute


class _Resp:
    def __init__(self, text):
        self.text = text


def test_analyze_dispute_parses_json():
    payload = {"ocr": {"printed_mrp_values": [90.0]}, "product_match": {"matches": False}}

    async def fake_gen(**kwargs):
        return _Resp(json.dumps(payload))

    with patch("app.services.dispute_service.generate_content_with_fallback", side_effect=fake_gen):
        out = asyncio.get_event_loop().run_until_complete(
            analyze_dispute(["data:image/jpeg;base64,AAAA"], "mrp_abuse", "Oil 1L", "overcharged")
        )
    assert out["product_match"]["matches"] is False


def test_analyze_dispute_empty_text_raises():
    async def fake_gen(**kwargs):
        return _Resp("")

    with patch("app.services.dispute_service.generate_content_with_fallback", side_effect=fake_gen):
        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(
                analyze_dispute(["data:image/jpeg;base64,AAAA"], None, "x", "y")
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_dispute_service.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Add `build_dispute_generation_config` to `app/services/gemini_service.py`**

Append (mirrors `build_claim_generation_config`):

```python
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
```

- [ ] **Step 4: Create `app/services/dispute_service.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_dispute_service.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/dispute_service.py app/services/gemini_service.py tests/test_dispute_service.py
git commit -m "feat(dispute): Gemini multi-image observation service"
```

---

### Task 10: Audit-service integration for the dispute endpoint

**Files:**
- Modify: `app/services/audit_service.py` (register `dispute` model, routing, downgrade, prompt version)
- Test: `tests/test_dispute_audit.py`

**Interfaces:**
- Consumes: `DisputeResponse` (Task 2).
- Produces: `audit_service` now handles `endpoint="dispute"` in `_RESPONSE_MODELS`, `_derive_routing`, `_downgrade`, `_prompt_version`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispute_audit.py
from app.models.dispute_response import DisputeResponse, RefundResult
from app.services.audit_service import _derive_routing, _downgrade


def _resp(decision="approve", route="auto"):
    return DisputeResponse(
        request_id="dsp_1", order_tracking_id="JM-1", category="mrp_abuse",
        category_source="provided", decision=decision, route=route,
        refund=RefundResult(eligible=True, amount=4.0, type="price_difference"),
        recommendation="ok", confidence=0.9, observations={}, model_used="m",
    )


def test_derive_routing_dispute_auto():
    action, status, prio, routed = _derive_routing("dispute", _resp())
    assert action == "approve"
    assert routed == "auto"


def test_derive_routing_dispute_agent():
    _, _, _, routed = _derive_routing("dispute", _resp(decision="agent", route="agent"))
    assert routed == "human"


def test_downgrade_forces_agent():
    out = _downgrade("dispute", _resp())
    assert out.decision == "agent"
    assert out.route == "agent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_dispute_audit.py -q`
Expected: FAIL (`KeyError: 'dispute'` / assertion in `_derive_routing`).

- [ ] **Step 3: Wire `dispute` into `app/services/audit_service.py`**

Add the import and register the model:

```python
from app.models.dispute_response import DisputeResponse
```

```python
_RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "scan": ScanResponse,
    "verify_claim": VerifyClaimResponse,
    "dispute": DisputeResponse,
}
```

In `_derive_routing`, before the final `verify_claim` assert block, add:

```python
    if endpoint == "dispute":
        assert isinstance(response, DisputeResponse)
        routed_to = "human" if response.route == "agent" else "auto"
        return response.decision, response.category, None, routed_to
```

In `_downgrade`, before the `note = ...` verify branch, add:

```python
    if endpoint == "dispute":
        return response.model_copy(update={
            "decision": "agent",
            "route": "agent",
            "agent_flags": list(getattr(response, "agent_flags", []) or []) + ["audit_write_failed"],
            "recommendation": (getattr(response, "recommendation", "") or "")
            + " [Audit write failed — forced agent review.]",
        })
```

In `_prompt_version`, replace the body with:

```python
def _prompt_version(endpoint: str) -> str:
    if endpoint == "scan":
        return settings.scan_prompt_version
    if endpoint == "dispute":
        return settings.dispute_prompt_version
    return settings.verify_prompt_version
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_dispute_audit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/audit_service.py tests/test_dispute_audit.py
git commit -m "feat(dispute): wire dispute endpoint into audit/idempotency"
```

---

### Task 11: Router + response assembly + main registration

**Files:**
- Create: `app/routers/dispute.py`
- Modify: `app/main.py` (include the router)
- Test: `tests/test_dispute_endpoint.py`

**Interfaces:**
- Consumes: everything above. Produces the `POST /api/v1/imgrecog/dispute` endpoint.
- Routing rule: `route = "agent"` when `decision == "agent"` OR `settings.dispute_assist_mode` OR category not in autonomous set; else `"auto"`. (The engine already converts gate hits to `decision="agent"`; assist/non-autonomous keep the recommendation but force `route="agent"`.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispute_endpoint.py
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
HEADERS = {"x-api-key": "test-secret"}


def _body(**over):
    base = {
        "images": ["data:image/jpeg;base64,AAAA"],
        "dispute_category": "mrp_abuse",
        "ticket": {"title": "t", "description": "overcharged", "notes": "", "disposition_code": ""},
        "shipment": {"order_tracking_id": "JM-1", "product_name": "Oil 1L", "product_type": "non_fnv",
                     "mrp": 100.0, "selling_price": 100.0, "invoice_amount": 100.0,
                     "quantity": 1, "seller_type": "1P"},
    }
    base.update(over)
    return base


def test_dispute_requires_api_key():
    assert client.post("/api/v1/imgrecog/dispute", json=_body()).status_code == 401


def test_dispute_mrp_approve():
    obs = {"ocr": {"printed_mrp_values": [90.0]}, "ai_generated": {"ai_probability": 0.0}}

    async def fake_analyze(*a, **k):
        return obs

    with patch("app.routers.dispute.analyze_dispute", side_effect=fake_analyze):
        r = client.post("/api/v1/imgrecog/dispute", json=_body(), headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "mrp_abuse"
    assert data["decision"] == "approve"
    assert data["refund"]["amount"] == 10.0


def test_dispute_insufficient_data_to_agent():
    body = _body(dispute_category=None,
                 ticket={"title": "", "description": "", "notes": "", "disposition_code": ""})

    async def fake_analyze(*a, **k):
        return {"ocr": {}, "ai_generated": {"ai_probability": 0.0}}

    with patch("app.routers.dispute.analyze_dispute", side_effect=fake_analyze):
        r = client.post("/api/v1/imgrecog/dispute", json=body, headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["decision"] == "agent"
    assert "insufficient_data" in data["agent_flags"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_dispute_endpoint.py -q`
Expected: FAIL (404 — route not registered).

- [ ] **Step 3: Create `app/routers/dispute.py`**

```python
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
from app.utils.image_utils import compute_image_phash, validate_image_size
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _autonomous() -> set:
    return {c.strip() for c in settings.dispute_autonomous_categories.split(",") if c.strip()}


def _request_id() -> str:
    return f"dsp_{int(time.time()*1000)}"


@router.post(
    "/api/v1/imgrecog/dispute",
    response_model=DisputeResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("100/minute")
async def dispute(request: Request, body: DisputeRequest) -> DisputeResponse:
    start = time.time()
    primary_image = body.images[0]

    try:
        validate_image_size(primary_image)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    # Idempotency (reuse the shared helper; image identity = primary image phash).
    image_phash = compute_image_phash(primary_image)
    idempotency_key = build_idempotency_key(
        body.idempotency_key, body.claim_id, body.shipment.order_tracking_id,
        "dispute", image_phash, primary_image,
    )
    replay = await find_replay("dispute", idempotency_key)
    if replay is not None:
        logger.info("idempotent_replay", endpoint="dispute",
                    order_id=body.shipment.order_tracking_id, request_id=replay.request_id)
        return replay

    # Category first — if unresolved, escalate without a model call.
    category, source = classify_category(body.dispute_category, body.ticket)
    if category is None:
        resp = _assemble(body, None, "none", {},
                         decision="agent", route="agent", flags=["insufficient_data"],
                         refund=RefundResult(), recommendation=(
                             "No category resolvable from description, notes, or disposition."),
                         confidence=0.0)
        return await _persist(request, body, primary_image, image_phash, idempotency_key, resp, {}, start)

    # One Gemini observation call.
    try:
        observations = await analyze_dispute(
            body.images, category, body.shipment.product_name,
            body.ticket.description or body.ticket.title,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Image analysis timed out")
    except ValueError as exc:
        logger.warning("dispute_analysis_failed", error=str(exc),
                       order_id=body.shipment.order_tracking_id)
        resp = _assemble(body, category, source, {}, decision="agent", route="agent",
                         flags=["low_confidence"], refund=RefundResult(),
                         recommendation="Image could not be analysed; route to agent.", confidence=0.0)
        return await _persist(request, body, primary_image, image_phash, idempotency_key, resp, {}, start)

    observations["_desc_len"] = len((body.ticket.description or "").strip())

    # Fraud signals (reuse dedup). AI prob comes from the model observation.
    dedup_result = await find_duplicates(image_phash, body.shipment.order_tracking_id, "dispute")
    signals = {
        "ai_probability": float((observations.get("ai_generated") or {}).get("ai_probability", 0.0)),
        "dedup_cross": dedup_result.is_cross_claim_duplicate,
        "web_hard": False,
    }

    d = decide(category, source, observations, body.shipment, body.is_rebuttal, signals)

    route = "agent" if (
        d.decision == "agent" or settings.dispute_assist_mode or category not in _autonomous()
    ) else "auto"

    resp = _assemble(
        body, category, source, observations,
        decision=d.decision, route=route, flags=d.agent_flags,
        refund=RefundResult(**d.refund), recommendation=d.recommendation, confidence=d.confidence,
    )
    out = await _persist(request, body, primary_image, image_phash, idempotency_key, resp,
                         observations, start)
    await register_image(image_phash, body.shipment.order_tracking_id, "dispute", out.request_id)
    logger.info("dispute_complete", request_id=out.request_id,
                order_id=body.shipment.order_tracking_id, category=category,
                decision=out.decision, route=out.route,
                latency_ms=round((time.time() - start) * 1000))
    return out


def _assemble(body, category, source, observations, *, decision, route, flags,
              refund, recommendation, confidence) -> DisputeResponse:
    # Strip the internal helper key before returning observations to the caller.
    obs = {k: v for k, v in (observations or {}).items() if not k.startswith("_")}
    return DisputeResponse(
        request_id=_request_id(),
        order_tracking_id=body.shipment.order_tracking_id,
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
        order_id=body.shipment.order_tracking_id, user_id="dispute",
        image_base64=primary_image, image_phash=image_phash, response=resp,
        latency_ms=latency_ms,
        raw_observations={k: v for k, v in (observations or {}).items() if not k.startswith("_")},
        computed={"decision": resp.decision, "route": resp.route,
                  "refund": resp.refund.model_dump(), "agent_flags": resp.agent_flags},
    )
```

- [ ] **Step 4: Register the router in `app/main.py`**

Find where `verify` router is included (e.g. `app.include_router(verify.router)`) and add alongside:

```python
from app.routers import dispute  # add to the existing routers import group
...
app.include_router(dispute.router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_dispute_endpoint.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite (no regressions)**

Run: `venv/bin/pytest -q`
Expected: PASS (existing 91 + new tests).

- [ ] **Step 7: Commit**

```bash
git add app/routers/dispute.py app/main.py tests/test_dispute_endpoint.py
git commit -m "feat(dispute): add /dispute endpoint and register router"
```

---

### Task 12: Alembic migration — dispute audit columns

**Files:**
- Create: `alembic/versions/0002_add_dispute_columns.py`
- Modify: `app/db/models.py` (add nullable columns)
- Test: `tests/test_dispute_audit_columns.py`

**Interfaces:**
- Produces: nullable columns on `claim_decisions` — `category`, `category_source`, `decision`, `route`, `agent_flags` (JSON), `refund` (JSON). Expand/contract: all nullable so old app versions keep working. `endpoint` check-constraint must allow `dispute`.

- [ ] **Step 1: Inspect the existing model + constraint**

Run: `venv/bin/python -c "import app.db.models as m; print([c.name for c in m.ClaimDecision.__table__.columns])"`
Expected: prints the current column list (confirm `endpoint` exists and note the check-constraint name in `app/db/models.py`).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_dispute_audit_columns.py
from app.db.models import ClaimDecision


def test_dispute_columns_exist():
    cols = {c.name for c in ClaimDecision.__table__.columns}
    assert {"category", "category_source", "decision", "route", "agent_flags", "refund"} <= cols
```

- [ ] **Step 3: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_dispute_audit_columns.py -q`
Expected: FAIL (columns missing).

- [ ] **Step 4: Add nullable columns to `app/db/models.py`**

In the `ClaimDecision` model, add (use the project's existing `Column`/`JSON` imports and `mapped_column` style — match the file):

```python
    category = Column(String, nullable=True)
    category_source = Column(String, nullable=True)
    decision = Column(String, nullable=True)
    route = Column(String, nullable=True)
    agent_flags = Column(JSON, nullable=True)
    refund = Column(JSON, nullable=True)
```

If the existing `endpoint` column has a CheckConstraint listing allowed values, add `"dispute"` to that list.

- [ ] **Step 5: Create `alembic/versions/0002_add_dispute_columns.py`**

```python
"""add dispute columns

Revision ID: 0002_add_dispute_columns
Revises: 0001_create_claim_decisions
"""
import sqlalchemy as sa
from alembic import op

revision = "0002_add_dispute_columns"
down_revision = "0001_create_claim_decisions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("claim_decisions", sa.Column("category", sa.String(), nullable=True))
    op.add_column("claim_decisions", sa.Column("category_source", sa.String(), nullable=True))
    op.add_column("claim_decisions", sa.Column("decision", sa.String(), nullable=True))
    op.add_column("claim_decisions", sa.Column("route", sa.String(), nullable=True))
    op.add_column("claim_decisions", sa.Column("agent_flags", sa.JSON(), nullable=True))
    op.add_column("claim_decisions", sa.Column("refund", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("claim_decisions", "refund")
    op.drop_column("claim_decisions", "agent_flags")
    op.drop_column("claim_decisions", "route")
    op.drop_column("claim_decisions", "decision")
    op.drop_column("claim_decisions", "category_source")
    op.drop_column("claim_decisions", "category")
```

> Note: persisting these columns into the row is optional for v1 — the full
> response snapshot already captures them. If `DecisionRecord`/`insert` are
> extended to populate the flat columns, do it here; otherwise the migration just
> makes them available for later querying/alerting. Keep `down_revision` matching
> the real revision id of `0001_create_claim_decisions.py`.

- [ ] **Step 6: Run test to verify it passes**

Run: `venv/bin/pytest tests/test_dispute_audit_columns.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add alembic/versions/0002_add_dispute_columns.py app/db/models.py tests/test_dispute_audit_columns.py
git commit -m "feat(dispute): add nullable dispute columns (expand/contract migration)"
```

---

### Task 13: Scalability hardening — Redis rate limit + Gemini concurrency semaphore

**Files:**
- Modify: `app/middleware/rate_limit.py` (Redis storage when `REDIS_URL` set)
- Modify: `app/services/gemini_service.py` (per-instance semaphore around model calls)
- Test: `tests/test_gemini_semaphore.py`

**Interfaces:**
- Produces: `generate_content_with_fallback` acquires a module-level `asyncio.Semaphore(settings.gemini_max_concurrency)` before calling the provider (backpressure); rate limiter uses Redis storage when configured (global limit across replicas).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gemini_semaphore.py
from app.services import gemini_service


def test_semaphore_sized_from_settings():
    sem = gemini_service._get_gemini_semaphore()
    assert sem._value == gemini_service.settings.gemini_max_concurrency
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_gemini_semaphore.py -q`
Expected: FAIL (`AttributeError: _get_gemini_semaphore`).

- [ ] **Step 3: Add the semaphore to `app/services/gemini_service.py`**

Near the top (after `_clients`):

```python
_gemini_semaphore: "asyncio.Semaphore | None" = None


def _get_gemini_semaphore() -> asyncio.Semaphore:
    global _gemini_semaphore
    if _gemini_semaphore is None:
        _gemini_semaphore = asyncio.Semaphore(settings.gemini_max_concurrency)
    return _gemini_semaphore
```

Wrap the provider call inside `generate_content_with_fallback` — change the first `try` so the primary call is guarded:

```python
    primary = "vertex" if settings.use_vertex else "api_key"
    async with _get_gemini_semaphore():
        try:
            return await get_client(primary).aio.models.generate_content(
                model=model, contents=contents, config=config,
            )
        except Exception as exc:  # noqa: BLE001
            if not _should_retry_with_vertex(exc, primary):
                raise
            logger.warning("gemini_retrying_with_vertex", primary_provider=primary,
                           code=getattr(exc, "code", None))
            try:
                return await get_client("vertex").aio.models.generate_content(
                    model=model, contents=contents, config=config,
                )
            except Exception as fallback_exc:  # noqa: BLE001
                logger.error("gemini_vertex_fallback_failed",
                             code=getattr(fallback_exc, "code", None), error=str(fallback_exc))
                raise
```

- [ ] **Step 4: Make the rate limiter Redis-backed when configured**

In `app/middleware/rate_limit.py`, where the `Limiter` is constructed, pass the Redis storage when set:

```python
from app.config.settings import settings

_storage_uri = settings.redis_url or "memory://"
limiter = Limiter(key_func=<existing key func>, storage_uri=_storage_uri)
```

(Keep the existing `key_func`; only add `storage_uri`. With `REDIS_URL` set, the 100/min limit is global across replicas.)

- [ ] **Step 5: Run test + full suite**

Run: `venv/bin/pytest tests/test_gemini_semaphore.py -q && venv/bin/pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/services/gemini_service.py app/middleware/rate_limit.py tests/test_gemini_semaphore.py
git commit -m "feat(dispute): gemini concurrency semaphore + redis-backed rate limit"
```

---

### Task 14: Docs — README + code guide + spec cross-link

**Files:**
- Modify: `README.md` (add a `/dispute` endpoint section)
- Modify: `docs/application-code-guide.md` (add dispute flow + new files to the map)

**Interfaces:** none (docs).

- [ ] **Step 1: Add a `/dispute` section to `README.md`**

After the `7c. URL-based image evaluation` section, add a `7f. Grocery dispute verification` section documenting: the endpoint, the request/response contract (copy from the spec §4), the 7 categories table, the MRP rule, the refund ceiling, and the new env vars (Task 1). Keep the prose style of the surrounding sections.

- [ ] **Step 2: Add the dispute flow to `docs/application-code-guide.md`**

Add a "Request Flow 3: Dispute Verification" subsection mirroring the existing flow write-ups, and add the new files (`routers/dispute.py`, `models/dispute_*`, `services/dispute_service.py`, `services/dispute_engine.py`, `services/category_classifier.py`) to the File-By-File Map.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/application-code-guide.md
git commit -m "docs(dispute): document /dispute endpoint and flow"
```

---

### Task 15: Production config guard for dispute money-vars

**Files:**
- Modify: `app/config/settings.py` (extend `_require_real_secrets_in_production` sanity checks)
- Test: `tests/test_dispute_prod_guard.py`

**Interfaces:**
- Produces: in production, invalid dispute numeric config (`refund_auto_approve_max < 0`, `dairy_min_shelf_pct` outside 0–100, `non_fnv_near_expiry_days < 0`) raises at boot.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispute_prod_guard.py
import pytest

from app.config.settings import Settings


def _prod(**over):
    base = dict(environment="production", kaily_api_secret="real", google_api_key="real",
                database_url="postgresql+asyncpg://x", object_store_provider="gcs",
                gcs_bucket="b", redis_url="redis://x")
    base.update(over)
    return base


def test_negative_refund_ceiling_rejected():
    with pytest.raises(ValueError):
        Settings(**_prod(refund_auto_approve_max=-1))


def test_bad_shelf_pct_rejected():
    with pytest.raises(ValueError):
        Settings(**_prod(dairy_min_shelf_pct=150))


def test_valid_prod_config_ok():
    Settings(**_prod())  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_dispute_prod_guard.py -q`
Expected: FAIL (no validation yet → bad values accepted).

- [ ] **Step 3: Extend the guard in `app/config/settings.py`**

Inside `_require_real_secrets_in_production`, before `if missing:`, add:

```python
        if self.refund_auto_approve_max < 0:
            missing.append("REFUND_AUTO_APPROVE_MAX (must be >= 0)")
        if not (0 <= self.dairy_min_shelf_pct <= 100):
            missing.append("DAIRY_MIN_SHELF_PCT (must be 0..100)")
        if self.non_fnv_near_expiry_days < 0:
            missing.append("NON_FNV_NEAR_EXPIRY_DAYS (must be >= 0)")
```

- [ ] **Step 4: Run test + full suite**

Run: `venv/bin/pytest tests/test_dispute_prod_guard.py -q && venv/bin/pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/config/settings.py tests/test_dispute_prod_guard.py
git commit -m "feat(dispute): production config guard for dispute money-vars"
```

---

## Self-Review

**Spec coverage:**
- §4 contract → Tasks 2, 11. §5 architecture → Tasks 9–11. §6 fallback chain → Task 3.
  §7 per-category logic → Tasks 6–8. §8 MRP refund math → Task 6. §9 escalation gates →
  Task 6 (`_apply_gates`) + Task 11 (route). §10.1 accuracy (determinism, prompt
  version, progressive autonomy) → Tasks 6–10, 1. §10.2 speed (one call, timeout,
  idempotent replay) → Tasks 9, 11. §10.3 scalability (Redis limit, semaphore) →
  Task 13. §10.4 production (audit, downgrade, guard, migration) → Tasks 10, 12, 15.
  §11 config → Task 1. §12 error handling → Tasks 9, 11. §13 testing → every task.
  §14 phasing → task order. §15 out-of-scope respected (no Shipment/Kapture calls).
- **Gaps consciously deferred (not blocking v1):** the eval harness (§10.1
  `tests/eval/`) and metrics/alerts (§10.4) are operational add-ons — add as a
  follow-up task set once the endpoint is live; image downscaling (§10.2) can reuse
  `image_url_fetcher`'s optimiser in a later task. Web-provenance signal is wired as
  `web_hard: False` placeholder in Task 11 (dedup is active); enable the existing
  `detect_web_provenance` concurrently in a follow-up if desired.

**Placeholder scan:** no "TBD/TODO/handle edge cases" steps; every code step shows
complete code. The one deliberate two-step fix (Task 6 Step 4 adjusting a test
expectation) is explicit with exact text.

**Type consistency:** `decide()` signature, `DisputeDecision` fields, `RefundResult`
keys, and the observation dict keys (`printed_mrp_values`, `product_match.matches`,
`quality.internal_defect`, `count.counted_units`, `spoilage.mold_or_visible_spoilage`)
are consistent across Tasks 2, 6–9, 11. `endpoint="dispute"` is consistent across
Tasks 10–12. `_today` seam name consistent (Tasks 7 def, 7 tests).

**Known integration check the implementer must confirm at Task 11/12:**
`build_idempotency_key`'s 4th arg is `user_id` — the router passes the literal
`"dispute"` there (no real user in the dispute contract); confirm that's acceptable
for idempotency keying, or thread a real user id if the caller supplies one.
Confirm the real revision id string in `0001_create_claim_decisions.py` for
`down_revision` in Task 12.
