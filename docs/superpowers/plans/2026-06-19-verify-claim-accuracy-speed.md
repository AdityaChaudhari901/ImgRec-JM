# /verify-claim Accuracy + Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close requirement 1b (web-downloaded photo detection) and make the existing `/verify-claim` pipeline faster, by adding a Google Vision reverse-image-search signal, running independent signals concurrently, hardening the Gemini JSON with a response schema, and surfacing the score out of 100.

**Architecture:** A new resilient `web_provenance` service calls Google Cloud Vision `WEB_DETECTION` and returns a structured result. The deterministic `authenticity_engine` fuses it (hard-reject on full matches across ≥2 distinct domains, soft penalty otherwise), mirroring the existing dedup hard-signal. The router fans out the independent network calls (Gemini ∥ Vision ∥ Sightengine) with `asyncio`, so the new signal adds ≈0 serial latency. The 0–1 score is also surfaced as `score_out_of_100`.

**Tech Stack:** Python 3.11, FastAPI, google-genai (Gemini/Vertex), google-cloud-vision, Pydantic v2, pytest + pytest-asyncio.

## Global Constraints

- Score range internal stays **0.0–1.0**; `score_out_of_100` is a derived int `round(score * 100)`. Thresholds unchanged.
- Every new external signal must **degrade gracefully**: any error/missing-creds → signal skipped (`checked=False`), request never 500s on the new path. Mirror `dedup_service`/`_sightengine`.
- The final authenticity decision is computed **in code** (`authenticity_engine`), never taken from a model.
- Heavy/3rd-party libs are **imported lazily** so the test suite runs without them (pattern already used for GCS/Redis).
- No new `Co-Authored-By` trailers in commits (project rule).
- Default `ai_detector_provider` stays `internal` (do NOT change; Sightengine is opt-in).
- Web-match default: hard-reject at `web_match_hard_min_domains = 2`.
- All config via `app/config/settings.py` env vars; document new ones in `.env.example`.

---

### Task 1: Settings + dependency for the web-provenance signal

**Files:**
- Modify: `requirements.txt`
- Modify: `app/config/settings.py:32-53` (claim-verification settings block)
- Modify: `.env.example`
- Test: `tests/test_settings_web.py` (create)

**Interfaces:**
- Produces: `settings.web_provenance_enabled: bool`, `settings.web_match_hard_min_domains: int`, `settings.web_match_soft_penalty: float`, `settings.web_match_penalty_cap: int`, `settings.vision_timeout_seconds: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_settings_web.py
from app.config.settings import settings


def test_web_provenance_defaults():
    assert settings.web_provenance_enabled is True
    assert settings.web_match_hard_min_domains == 2
    assert settings.web_match_soft_penalty == 0.15
    assert settings.web_match_penalty_cap == 3
    assert settings.vision_timeout_seconds == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_settings_web.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'web_provenance_enabled'`

- [ ] **Step 3: Add the settings**

In `app/config/settings.py`, immediately after the `authenticity_review_threshold` line (currently line 53), add:

```python

    # ---- Web reverse-image-search (req 1b): "website-downloaded" detection ---
    # Google Cloud Vision WEB_DETECTION. Off -> signal skipped (checked=False).
    web_provenance_enabled: bool = True
    # Full matches across at least this many DISTINCT domains -> hard fraud signal
    # (auto-reject, like a cross-claim duplicate). Raise high to disable hard-reject.
    web_match_hard_min_domains: int = 2
    # Soft, proportional score penalty per web match below the hard threshold.
    web_match_soft_penalty: float = 0.15
    # Max matches counted toward the soft penalty.
    web_match_penalty_cap: int = 3
    # Hard timeout (seconds) for the Vision call.
    vision_timeout_seconds: int = 8
```

- [ ] **Step 4: Add the dependency**

In `requirements.txt`, under the Phase 2 redis block, append:

```
# --- Phase 3: web reverse-image-search (req 1b) --------------------------
# Google Cloud Vision WEB_DETECTION. Imported lazily; the test suite mocks the
# client and never needs it installed.
google-cloud-vision==3.7.4
```

- [ ] **Step 5: Document env vars**

In `.env.example`, add (near the authenticity settings):

```
# Web reverse-image-search (req 1b) — "website-downloaded" detection
WEB_PROVENANCE_ENABLED=true
WEB_MATCH_HARD_MIN_DOMAINS=2
WEB_MATCH_SOFT_PENALTY=0.15
WEB_MATCH_PENALTY_CAP=3
VISION_TIMEOUT_SECONDS=8
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_settings_web.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt app/config/settings.py .env.example tests/test_settings_web.py
git commit -m "feat(verify): add web-provenance settings and Vision dependency"
```

---

### Task 2: `web_provenance` service (Vision client + resilient detection)

**Files:**
- Create: `app/services/web_provenance.py`
- Test: `tests/test_web_provenance.py` (create)

**Interfaces:**
- Consumes: `settings.vision_timeout_seconds`, `settings.web_provenance_enabled`.
- Produces:
  - `WebProvenanceResult` dataclass with fields `full_match_count: int`, `partial_match_count: int`, `distinct_pages: int`, `distinct_domains: int`, `best_guess_label: Optional[str]`, `checked: bool`.
  - `async def detect_web_provenance(image_base64: str) -> WebProvenanceResult`
  - `reset_vision_client() -> None` (test isolation)
  - `_count_distinct_domains(urls: list[str]) -> int` (helper, unit-tested)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_web_provenance.py
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.services import web_provenance
from app.services.web_provenance import (
    WebProvenanceResult,
    _count_distinct_domains,
    detect_web_provenance,
    reset_vision_client,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_vision_client()
    yield
    reset_vision_client()


def test_count_distinct_domains_strips_www_and_dedupes():
    urls = [
        "https://www.example.com/a.jpg",
        "https://example.com/b.jpg",          # same domain as above
        "http://shop.other.com/x",
        "not a url",                            # ignored
    ]
    assert _count_distinct_domains(urls) == 2


def _fake_web_detection(full=0, partial=0, pages=None, label="a shoe"):
    pages = pages or []
    wd = SimpleNamespace(
        full_matching_images=[SimpleNamespace(url=f"https://d{i}.com/i.jpg") for i in range(full)],
        partial_matching_images=[SimpleNamespace(url=f"https://p{i}.com/i.jpg") for i in range(partial)],
        pages_with_matching_images=[SimpleNamespace(url=u) for u in pages],
        best_guess_labels=[SimpleNamespace(label=label)] if label else [],
    )
    return SimpleNamespace(web_detection=wd, error=SimpleNamespace(message=""))


@pytest.mark.asyncio
async def test_full_matches_on_multiple_domains():
    fake = _fake_web_detection(
        full=2, pages=["https://a.com/p", "https://b.com/p"], label="cracked phone"
    )
    with patch.object(web_provenance, "_get_vision_client") as gc:
        gc.return_value.web_detection.return_value = fake
        result = await detect_web_provenance("data:image/jpeg;base64,/9j/fake")
    assert result.checked is True
    assert result.full_match_count == 2
    assert result.distinct_domains == 2
    assert result.best_guess_label == "cracked phone"


@pytest.mark.asyncio
async def test_clean_image_has_no_matches():
    with patch.object(web_provenance, "_get_vision_client") as gc:
        gc.return_value.web_detection.return_value = _fake_web_detection()
        result = await detect_web_provenance("data:image/jpeg;base64,/9j/fake")
    assert result.checked is True
    assert result.full_match_count == 0
    assert result.distinct_domains == 0


@pytest.mark.asyncio
async def test_vision_error_degrades_to_unchecked():
    with patch.object(web_provenance, "_get_vision_client", side_effect=RuntimeError("boom")):
        result = await detect_web_provenance("data:image/jpeg;base64,/9j/fake")
    assert result.checked is False
    assert result.full_match_count == 0


@pytest.mark.asyncio
async def test_bad_base64_degrades_to_unchecked():
    result = await detect_web_provenance("not-base64-@@@")
    assert isinstance(result, WebProvenanceResult)
    assert result.checked is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web_provenance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.web_provenance'`

- [ ] **Step 3: Implement the service**

```python
# app/services/web_provenance.py
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
    from google.cloud import vision  # lazy import

    client = _get_vision_client()
    response = client.web_detection(image=vision.Image(content=image_bytes))
    if getattr(response, "error", None) and getattr(response.error, "message", ""):
        logger.error("vision_api_error", error=response.error.message)
        return WebProvenanceResult(checked=False)

    wd = response.web_detection
    full = list(getattr(wd, "full_matching_images", []) or [])
    partial = list(getattr(wd, "partial_matching_images", []) or [])
    pages = list(getattr(wd, "pages_with_matching_images", []) or [])
    labels = list(getattr(wd, "best_guess_labels", []) or [])

    domain_urls = [i.url for i in full] + [p.url for p in pages]
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web_provenance.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/web_provenance.py tests/test_web_provenance.py
git commit -m "feat(verify): add resilient Google Vision web-provenance service"
```

---

### Task 3: `WebProvenanceCheck` response model

**Files:**
- Modify: `app/models/verify_response.py:37-63`
- Test: `tests/test_verify_response_model.py` (create)

**Interfaces:**
- Produces: `WebProvenanceCheck(BaseModel)` with `checked: bool`, `full_matches: int`, `partial_matches: int`, `distinct_domains: int`, `reason: Optional[str]`. Added as `AuthenticityChecks.web_provenance` (optional, default `None`). `VerifyClaimResponse.score_out_of_100: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify_response_model.py
from app.models.verify_response import AuthenticityChecks, WebProvenanceCheck


def test_web_provenance_check_defaults():
    c = WebProvenanceCheck(checked=False)
    assert c.full_matches == 0
    assert c.distinct_domains == 0
    assert c.reason is None


def test_authenticity_checks_web_provenance_optional():
    # web_provenance defaults to None so existing callers are unaffected.
    assert "web_provenance" in AuthenticityChecks.model_fields
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_verify_response_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'WebProvenanceCheck'`

- [ ] **Step 3: Add the model fields**

In `app/models/verify_response.py`, add this class immediately before `class AuthenticityChecks` (currently line 37):

```python
class WebProvenanceCheck(BaseModel):
    checked: bool  # False when Vision was skipped/unavailable
    full_matches: int = 0
    partial_matches: int = 0
    distinct_domains: int = 0
    reason: Optional[str] = None


```

Then add a field to `AuthenticityChecks` (after `product_match: ProductMatchCheck`):

```python
    web_provenance: Optional[WebProvenanceCheck] = None
```

Then add a field to `VerifyClaimResponse` immediately after `authenticity_score: float  # 0..1`:

```python
    score_out_of_100: int  # authenticity_score surfaced as 0..100 (req 3)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_verify_response_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/verify_response.py tests/test_verify_response_model.py
git commit -m "feat(verify): add WebProvenanceCheck and score_out_of_100 to response"
```

---

### Task 4: Fuse web signal into the scoring engine + surface /100

**Files:**
- Modify: `app/services/authenticity_engine.py:60-233`
- Test: `tests/test_authenticity_engine.py` (extend)

**Interfaces:**
- Consumes: `WebProvenanceResult` (Task 2), `WebProvenanceCheck` (Task 3), `settings.web_match_hard_min_domains`, `settings.web_match_soft_penalty`, `settings.web_match_penalty_cap`.
- Produces: `score_claim(gemini, ai_check, dedup_result=None, web_result=None)` now returns checks including `web_provenance`; `build_verify_response(..., dedup_result=None, web_result=None)` sets `score_out_of_100` and `checks.web_provenance`. New helper `_web_hard(web_result) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_authenticity_engine.py
from app.services.web_provenance import WebProvenanceResult


def _web(full=0, partial=0, domains=0, checked=True):
    return WebProvenanceResult(
        full_match_count=full, partial_match_count=partial,
        distinct_domains=domains, checked=checked,
    )


def test_web_match_two_domains_forces_reject():
    score, verdict, action, checks = score_claim(
        _gemini(0.99, True, 0.99, True), NO_AI, web_result=_web(full=2, domains=2)
    )
    assert verdict == "likely_fraud"
    assert action == "reject"
    assert score == 0.0
    assert checks.web_provenance is not None and checks.web_provenance.distinct_domains == 2


def test_web_match_single_domain_is_soft_penalty_only():
    clean = score_claim(_gemini(0.9, True, 0.9, True), NO_AI, web_result=_web())[0]
    soft = score_claim(_gemini(0.9, True, 0.9, True), NO_AI, web_result=_web(full=1, domains=1))[0]
    assert soft < clean
    assert soft > 0.0  # not a hard reject


def test_web_unchecked_leaves_score_unchanged():
    base = score_claim(_gemini(0.9, True, 0.9, True), NO_AI)[0]
    unchecked = score_claim(
        _gemini(0.9, True, 0.9, True), NO_AI, web_result=_web(checked=False)
    )[0]
    assert unchecked == base


def test_score_out_of_100_matches_authenticity_score():
    resp = build_verify_response(_gemini(0.9, True, 0.9, True), NO_AI, "JM-1", "u_1")
    assert resp.score_out_of_100 == round(resp.authenticity_score * 100)


def test_web_hard_decision_confidence_is_max():
    resp = build_verify_response(
        _gemini(0.99, True, 0.99, True), NO_AI, "JM-1", "u_1", web_result=_web(full=3, domains=3)
    )
    assert resp.recommended_action == "reject"
    assert resp.decision_confidence == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_authenticity_engine.py -v`
Expected: FAIL (`score_claim() got an unexpected keyword argument 'web_result'`)

- [ ] **Step 3: Update imports and add the web helper**

In `app/services/authenticity_engine.py`, add to the imports near the `DedupResult` import:

```python
from app.services.web_provenance import WebProvenanceResult
from app.models.verify_response import WebProvenanceCheck
```

(Add `WebProvenanceCheck` to the existing `from app.models.verify_response import (...)` group instead of a second line if you prefer; keep one import block.)

Add this helper after `_f`:

```python
def _web_hard(web_result: Optional[WebProvenanceResult]) -> bool:
    """A confirmed full match across enough distinct domains is a hard fraud
    signal — a genuine damage photo does not live on multiple unrelated sites."""
    return bool(
        web_result
        and web_result.checked
        and web_result.full_match_count > 0
        and web_result.distinct_domains >= settings.web_match_hard_min_domains
    )
```

- [ ] **Step 4: Thread `web_result` through `score_claim`**

Change the `score_claim` signature and body. New signature:

```python
def score_claim(
    gemini: dict,
    ai_check: AIGeneratedCheck,
    dedup_result: Optional[DedupResult] = None,
    web_result: Optional[WebProvenanceResult] = None,
) -> Tuple[float, str, str, AuthenticityChecks]:
```

After the existing `hard_duplicate` block (the one that appends the duplicate flag) and BEFORE `checks = AuthenticityChecks(...)`, add the web-provenance check object + hard flag:

```python
    web_hard = _web_hard(web_result)
    web_check = None
    if web_result is not None and web_result.checked:
        reason = (
            f"image found on {web_result.distinct_domains} domain(s), "
            f"{web_result.full_match_count} full match(es)"
        )
        web_check = WebProvenanceCheck(
            checked=True,
            full_matches=web_result.full_match_count,
            partial_matches=web_result.partial_match_count,
            distinct_domains=web_result.distinct_domains,
            reason=reason,
        )
        if web_hard:
            other_flags.append(f"web_download_match: {reason}")
    elif web_result is not None:
        web_check = WebProvenanceCheck(checked=False)
```

Update the `AuthenticityChecks(...)` construction to include the new field:

```python
    checks = AuthenticityChecks(
        ai_generated=ai_check,
        image_comment_alignment=alignment,
        product_match=product,
        other_flags=other_flags,
        web_provenance=web_check,
    )
```

After the existing `score -= settings.authenticity_flag_penalty * len(other_flags)` line and BEFORE `score = _clamp(score)`, add the soft web penalty (only when not a hard match):

```python
    if web_result is not None and web_result.checked and not web_hard:
        web_hits = min(
            web_result.full_match_count + web_result.partial_match_count,
            settings.web_match_penalty_cap,
        )
        score -= settings.web_match_soft_penalty * web_hits
```

Update the hard-zero line and the routing call to include `web_hard`:

```python
    if hard_duplicate or web_hard:
        score = 0.0  # a confirmed reuse / web-download is inauthentic by definition

    verdict, action = _route(score, ai_confident, ai_check.source, hard_duplicate or web_hard)
    return round(score, 3), verdict, action, checks
```

- [ ] **Step 5: Thread `web_result` through `build_verify_response` and set `score_out_of_100`**

Change the `build_verify_response` signature:

```python
def build_verify_response(
    gemini: dict,
    ai_check: AIGeneratedCheck,
    order_id: str,
    user_id: str,
    dedup_result: Optional[DedupResult] = None,
    web_result: Optional[WebProvenanceResult] = None,
) -> VerifyClaimResponse:
```

Update its body: pass `web_result` to `score_claim`, fold `web_hard` into the hard-signal confidence, and add `score_out_of_100`:

```python
    score, verdict, action, checks = score_claim(gemini, ai_check, dedup_result, web_result)
    ai_confident = (
        ai_check.is_ai_generated
        and ai_check.ai_probability >= settings.ai_detection_min_confidence
    )
    hard_signal = bool(dedup_result and dedup_result.is_cross_claim_duplicate) or _web_hard(web_result)
    conf = 1.0 if hard_signal else decision_confidence(score, action, ai_check, ai_confident)
    return VerifyClaimResponse(
        success=True,
        request_id=generate_request_id(),
        order_id=order_id,
        user_id=user_id,
        authenticity_score=score,
        score_out_of_100=round(score * 100),
        decision_confidence=conf,
        verdict=verdict,  # type: ignore[arg-type]
        recommended_action=action,  # type: ignore[arg-type]
        recognition=_recognition(gemini),
        checks=checks,
        agent_comment=_agent_comment(gemini, checks, score, verdict, action),
        processed_at=datetime.now(timezone.utc),
        model_used=settings.gemini_model,
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_authenticity_engine.py -v`
Expected: PASS (all existing + 5 new). The existing `test_build_verify_response_shape` still passes because `web_result` defaults to `None`.

- [ ] **Step 7: Commit**

```bash
git add app/services/authenticity_engine.py tests/test_authenticity_engine.py
git commit -m "feat(verify): fuse web-provenance signal into scoring + score_out_of_100"
```

---

### Task 5: Parallel fan-out in the router + audit the web signal

**Files:**
- Modify: `app/routers/verify.py:1-128`
- Test: `tests/test_verify.py` (extend)

**Interfaces:**
- Consumes: `detect_web_provenance` (Task 2), updated `build_verify_response` (Task 4), `settings.web_provenance_enabled`, `settings.ai_detector_provider`.
- Produces: response with `score_out_of_100` and `checks.web_provenance`; the audit `computed` dict gains a `web_provenance` entry.

- [ ] **Step 1: Write the failing test**

Match the existing style in `tests/test_verify.py` exactly: async tests with `AsyncClient`/`ASGITransport`, the module-level `HEADERS` and `PAYLOAD` constants, and `patch(..., return_value=...)` (patch auto-detects the async target and returns an AsyncMock). Append:

```python
# append to tests/test_verify.py
from app.services.web_provenance import WebProvenanceResult

_WEB_CLEAN = WebProvenanceResult(checked=True, full_match_count=0, distinct_domains=0)
_WEB_STOLEN = WebProvenanceResult(checked=True, full_match_count=3, distinct_domains=3)


@pytest.mark.asyncio
async def test_verify_includes_score_out_of_100_and_web_check():
    with patch("app.routers.verify.analyze_claim", return_value=GEMINI_AUTHENTIC), \
         patch("app.routers.verify.detect_web_provenance", return_value=_WEB_CLEAN):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/v1/imgrecog/verify-claim", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["score_out_of_100"] == round(body["authenticity_score"] * 100)
    assert body["checks"]["web_provenance"]["checked"] is True


@pytest.mark.asyncio
async def test_verify_web_download_match_rejects():
    with patch("app.routers.verify.analyze_claim", return_value=GEMINI_AUTHENTIC), \
         patch("app.routers.verify.detect_web_provenance", return_value=_WEB_STOLEN):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/v1/imgrecog/verify-claim", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_action"] == "reject"
    assert body["score_out_of_100"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_verify.py -v`
Expected: FAIL (response has no `score_out_of_100` / `detect_web_provenance` not patchable in `verify` module yet)

- [ ] **Step 3: Update the router imports**

In `app/routers/verify.py`, add to the imports block:

```python
import asyncio

from app.services.web_provenance import detect_web_provenance, WebProvenanceResult
```

(`asyncio` import goes at the top with the stdlib imports; the service import goes with the other `app.services` imports.)

- [ ] **Step 4: Fan out the independent calls**

Replace the current sequential block (from the `# Gemini observations` comment through the `dedup_result = await find_duplicates(...)` line and the `build_verify_response(...)` call) with the concurrent version:

```python
    # Fan out the independent network calls. Web reverse-search and (when enabled)
    # the Sightengine detector don't depend on Gemini, so they run concurrently —
    # total latency ~= the Gemini call alone instead of the serial sum.
    web_task = (
        asyncio.create_task(detect_web_provenance(body.image_base64))
        if settings.web_provenance_enabled
        else None
    )
    ai_task = (
        asyncio.create_task(detect_ai_generated(body.image_base64, None))
        if settings.ai_detector_provider == "sightengine"
        else None
    )

    # Gemini observations — map upstream conditions to honest status codes.
    try:
        gemini = await analyze_claim(
            body.image_base64,
            body.user_comment,
            body.claimed_product,
            body.reference_image_base64,
        )
    except TimeoutError:
        if web_task:
            web_task.cancel()
        if ai_task:
            ai_task.cancel()
        raise HTTPException(status_code=504, detail="Image analysis timed out")
    except HTTPException:
        if web_task:
            web_task.cancel()
        if ai_task:
            ai_task.cancel()
        raise
    except _UPSTREAM_API_ERROR as exc:  # type: ignore[misc]
        if web_task:
            web_task.cancel()
        if ai_task:
            ai_task.cancel()
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
        if web_task:
            web_task.cancel()
        if ai_task:
            ai_task.cancel()
        logger.error("verify_failed", error=str(exc), order_id=body.order_id)
        raise HTTPException(status_code=500, detail="Claim analysis failed")

    # AI-generated check: Sightengine ran in parallel; internal needs Gemini's hint.
    if ai_task is not None:
        ai_check = await ai_task
    else:
        ai_check = await detect_ai_generated(body.image_base64, gemini.get("ai_generated", {}))

    # Reused-image fraud check (hard signal) — see dedup_service.
    dedup_result = await find_duplicates(image_phash, body.order_id, body.user_id)

    # Web reverse-search result (resilient — never raises).
    web_result = await web_task if web_task is not None else WebProvenanceResult(checked=False)

    response = build_verify_response(
        gemini, ai_check, body.order_id, body.user_id, dedup_result, web_result
    )
```

> Remove the now-duplicated original `ai_check = await detect_ai_generated(...)`, `dedup_result = ...`, and `build_verify_response(...)` lines that this block replaces.

- [ ] **Step 5: Add the web signal to the audit record**

In the `persist_decision(...)` call, extend the `computed=` dict:

```python
        computed={
            "authenticity_score": response.authenticity_score,
            "decision_confidence": response.decision_confidence,
            "dedup": dedup_result.to_audit(),
            "web_provenance": web_result.to_audit(),
        },
```

- [ ] **Step 6: Add the web flag to the completion log**

In the `logger.info("verify_complete", ...)` call, add:

```python
        web_full_matches=web_result.full_match_count,
        web_checked=web_result.checked,
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_verify.py tests/test_verify_dedup.py -v`
Expected: PASS (existing verify + dedup tests unaffected; 2 new pass)

- [ ] **Step 8: Commit**

```bash
git add app/routers/verify.py tests/test_verify.py
git commit -m "feat(verify): parallel signal fan-out + audit web-provenance"
```

---

### Task 6: Schema-lock the claim Gemini JSON

**Files:**
- Modify: `app/services/gemini_service.py:21-39` (config) and `app/services/claim_service.py:110-119` (use a claim-specific config)
- Test: `tests/test_claim_schema.py` (create)

**Interfaces:**
- Consumes: existing `build_generation_config`.
- Produces: `build_claim_generation_config() -> types.GenerateContentConfig` (adds a `response_schema` matching the claim JSON) in `gemini_service`. `claim_service.analyze_claim` uses it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_claim_schema.py
from app.services.gemini_service import build_claim_generation_config


def test_claim_config_has_json_mime_and_schema():
    cfg = build_claim_generation_config()
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_schema is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_claim_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_claim_generation_config'`

- [ ] **Step 3: Add the schema-bound config**

In `app/services/gemini_service.py`, after `build_generation_config`, add:

```python
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
```

- [ ] **Step 4: Use it in `claim_service`**

In `app/services/claim_service.py`, change the import:

```python
from app.services.gemini_service import build_claim_generation_config, get_client
```

and the config line inside `analyze_claim`:

```python
    config = build_claim_generation_config()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_claim_schema.py -v`
Expected: PASS

- [ ] **Step 6: Run the whole suite (no regressions)**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/services/gemini_service.py app/services/claim_service.py tests/test_claim_schema.py
git commit -m "feat(verify): schema-lock claim Gemini JSON for reliability"
```

---

## Self-Review

**Spec coverage:**
- Req 1a (AI-generated) — already works; Sightengine opt-in documented (no task needed; default unchanged per Global Constraints). ✓
- Req 1b (web-downloaded) — Tasks 2 (service), 3 (model), 4 (fusion), 5 (router). ✓
- Req 2 (product match) — already works; reliability hardened in Task 6 (response_schema). ✓
- Req 3 (score out of 100) — Task 3 (field) + Task 4 (compute). ✓
- Speed/parallelism — Task 5 (asyncio fan-out). ✓
- Settings + graceful degradation — Task 1 + resilient service in Task 2. ✓
- Audit trail of web signal — Task 5 Step 5. ✓
- pHash analysis cache — explicitly out of scope (spec §4); no task. ✓

**Placeholder scan:** No TBD/TODO. Task 5's tests use the verified real conventions of `tests/test_verify.py` (`HEADERS`/`PAYLOAD` constants, `AsyncClient`/`ASGITransport`, `GEMINI_AUTHENTIC`, `patch(..., return_value=...)`).

**Type consistency:** `WebProvenanceResult` fields (`full_match_count`, `partial_match_count`, `distinct_domains`, `checked`) are used identically across Tasks 2, 4, 5. `WebProvenanceCheck` fields (`full_matches`, `partial_matches`, `distinct_domains`, `checked`, `reason`) consistent across Tasks 3–4. `score_claim`/`build_verify_response` extended signatures match between Tasks 4 and 5. `detect_web_provenance` async signature consistent (Tasks 2, 5).
