# Design — `/verify-claim` accuracy + speed pass

**Date:** 2026-06-19
**Status:** Draft for review
**Endpoint affected:** `POST /api/v1/imgrecog/verify-claim`
**Goal (user's words):** *"best accuracy and fast"* against the three requirements:

1. Check if the image is **AI-generated OR website-downloaded**.
2. **Product image matches** the product name details.
3. `user_query + image_analysis = score out of 100`.

---

## 1. Context — what already exists

The pipeline is ~90% built. Mapping requirements to current code:

| Requirement | Current state |
|---|---|
| 1a. AI-generated | `ai_image_detector.detect_ai_generated` — `internal` (Gemini self-hint + EXIF/C2PA metadata) or `sightengine`. Advisory, correctly gated (only `sightengine` may auto-reject). |
| 1b. Website-downloaded | **Gap.** `dedup_service` only catches reuse *within our own prior claims* (pHash). No public-web reverse search. |
| 2. Product match | `claim_service` Gemini call → `product_match`; deterministic in `authenticity_engine`. Works. |
| 3. user_query + image → score | `user_query` **is** `user_comment` (claim-consistency, not search relevance). Fused into `authenticity_score` (0–1). Works, but **not surfaced as /100**. |

`user_query` ambiguity is resolved by the code: `VerifyClaimRequest.user_comment` = *"what issue the customer says they are facing"*.

## 2. The core tension and its resolution

"Best accuracy" demands a new signal (web reverse-search = a network call). "Fast" resists added latency. **Resolution: fan-out.** All signals that don't depend on each other run concurrently via `asyncio.gather`, so total latency ≈ `max(calls)` (dominated by the Gemini call) rather than `sum(calls)`. Adding reverse-search then costs ≈0 serial latency.

## 3. Scope (the four changes)

### 3.1 Web-downloaded detection (req 1b) — NEW
New service `app/services/web_provenance.py` using **Google Cloud Vision `WEB_DETECTION`** (project `alien-slice-499511-f8`, already enabled; ADC already authenticates).

- **Input:** image bytes. **Output:** a `WebProvenanceResult` dataclass:
  - `full_match_count` — exact/near-exact copies found on the web
  - `partial_match_count` — partial matches
  - `distinct_pages`, `distinct_domains` — breadth of where it appears
  - `best_guess_label` — Vision's guess at the subject (debug/audit only)
  - `checked: bool` — False when Vision was skipped/unavailable (degrade signal)
- **Resilience:** mirrors `dedup`/`sightengine` — any error (API off, no creds, timeout) → `checked=False`, empty result, **never raises**. Dev without Vision still works.
- **Client:** lazily-created singleton (mirror `gemini_service.get_client`); reuse Vertex project/ADC. Wrapped in `asyncio.wait_for` with its own timeout.

**Fusion** (in `authenticity_engine`, configurable, mirrors existing dedup/AI logic):
- A claimant's genuine damage photo should not exist on the public web. So:
  - `full_match_count >= web_match_hard_min_domains` **across distinct domains** (default **2**) → **hard signal**: cap score → `likely_fraud` / `reject`, with an auditable `other_flag` listing the pages. (Mirrors how a cross-claim duplicate hard-rejects.)
  - Otherwise, `score -= web_match_soft_penalty * min(full_match_count + partial_match_count, web_match_penalty_cap)` — a soft, proportional dent.
- **Decided:** **hard-reject at ≥2 distinct domains** (a damage photo on 2+ unrelated sites is lifted, not coincidence), soft proportional penalty below that. Rationale: genuine damage photos essentially never appear on multiple unrelated public pages; the false-positive case (customer posted their own photo to 2+ sites before claiming) is vanishingly rare, and a single-domain match stays soft. Configurable via `web_match_hard_min_domains` (raise it to effectively disable hard-reject).

### 3.2 AI-generated accuracy (req 1a) — opt-in upgrade
The `internal` detector leans on Gemini judging *itself*, which the market study shows is unreliable. `sightengine` is the accurate path and is **already coded + gated** (only `sightengine` may auto-reject).

**Do NOT change the default.** Keeping `ai_detector_provider=internal` avoids (a) forcing a paid Sightengine account you may not have and (b) the prod boot validator hard-failing when provider is `sightengine` without keys. Instead: document the one-line flip (`AI_DETECTOR_PROVIDER=sightengine` + keys) as the recommended accuracy upgrade once an account exists. Req 1a already functions on `internal` today; this is a quality dial, not a gap. **No code change.**

### 3.3 Product-match reliability (req 2)
Add a typed **`response_schema`** to the Gemini generation config for the claim call so the JSON shape can't drift and won't 500 on malformed output. `response_mime_type` is already set; this adds the schema. Keeps existing fields identical.

### 3.4 Score out of 100 (req 3)
Add `score_out_of_100: int` to `VerifyClaimResponse` = `round(authenticity_score * 100)`. Internal 0–1 `authenticity_score` and all thresholds stay unchanged (auditability preserved).

## 4. Latency design (the "fast" half)

Restructure `verify_claim` so independent work runs concurrently:

```
phash + dedup lookup ─┐  (fast, local/Redis)
Gemini claim analysis ─┤── asyncio.gather ──▶ fuse ─▶ score ─▶ persist
Vision web-detection ──┤
Sightengine (if on) ───┘
```

- `web_provenance` and `sightengine` are independent of Gemini → always parallel.
- `internal` AI detector needs Gemini's hint → in internal mode it runs after Gemini (cheap, local, no added network).
- Net: with Sightengine + Vision on, latency ≈ the single Gemini call instead of the current serial sum.
- **Optional (volume-only):** pHash cache for the two *image-only* signals (web-detection + AI-detector) so re-uploads skip paid calls. NOT applicable to the Gemini claim analysis (depends on `user_comment`/`claimed_product`, not reusable across claims). Deferred unless volume warrants.

## 5. New settings (`app/config/settings.py`)

| Setting | Default | Purpose |
|---|---|---|
| `web_provenance_enabled` | `true` | Master switch; off → signal skipped (`checked=False`). |
| `web_match_hard_min_domains` | `2` | Distinct domains with a full match to trigger hard reject. Raise to disable hard-reject. |
| `web_match_soft_penalty` | `0.15` | Per-match score penalty below the hard threshold. |
| `web_match_penalty_cap` | `3` | Max matches counted toward the soft penalty. |
| `vision_timeout_seconds` | `8` | Hard timeout for the Vision call. |
| `ai_detector_provider` | `internal` *(unchanged)* | Flip to `sightengine` (+ keys) as an opt-in accuracy upgrade; see 3.2. |

Production boot validation (`_require_real_secrets_in_production`) extended: if `web_provenance_enabled` and `use_vertex` is false, require explicit Vision credentials (when Vertex is on, ADC covers it).

## 6. Response shape change

`VerifyClaimResponse` gains:
- `score_out_of_100: int`
- `checks.web_provenance: WebProvenanceCheck` (`{full_matches, partial_matches, distinct_domains, checked, reason}`)

All existing fields unchanged → backward compatible for current Kaily consumers.

## 7. Testing

- **`web_provenance` unit tests:** mock the Vision client — full-match-heavy, partial-only, clean, and API-error (→ `checked=False`) cases.
- **`authenticity_engine` fusion tests:** hard-reject at ≥2 domains; soft penalty below; web signal absent (`checked=False`) leaves score unchanged; interaction with existing AI/dedup gates.
- **Parallelism:** a test asserting the independent calls are gathered (e.g. timing or call-order assertion via mocks).
- **`/100` surfacing:** `score_out_of_100 == round(authenticity_score * 100)` across a range.
- **`response_schema`:** existing claim-analysis tests still pass; malformed-output path covered.
- Run the full existing `tests/` suite — no regressions.

## 8. Out of scope (explicitly)

- Second-VLM cross-check / Hive (Approach C) — no labeled data to justify; diminishing returns.
- Downgrading to `gemini-flash-lite` — weaker on attribute mismatches; conflicts with "best accuracy".
- Changes to the `/scan` endpoint — requirements are about `/verify-claim`.
- pHash analysis cache for the Gemini call — not reusable across claims.

## 9. Prerequisites (status)

- Cloud Vision API enabled on `alien-slice-499511-f8` — **DONE**.
- ADC configured (`gcloud auth application-default login`) — **DONE** (token verified).
- Sightengine account + keys (`SIGHTENGINE_API_USER/SECRET`) — **optional** accuracy upgrade (3.2); absent → internal detector runs, so not a blocker to ship.
