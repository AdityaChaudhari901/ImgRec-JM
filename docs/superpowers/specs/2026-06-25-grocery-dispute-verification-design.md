# AI-Driven Grocery Image Verification — Design Spec

**Date:** 2026-06-25
**Status:** Draft for review
**Author:** Aditya Chaudhari (with Claude)
**Target:** JioMart / Kaily (Kapture CRM) dispute resolution — production
**Non-functional bar:** accurate (≥95%), fast (<30s), scalable (linear horizontal), production-ready

---

## 1. Summary

Add a **single new endpoint**, `POST /api/v1/imgrecog/dispute`, to the existing
`imgrecog-kaily` FastAPI service. It accepts everything needed to resolve a
grocery delivery dispute in **one request** — customer images, ticket text, and
shipment data passed in by the caller (Kaily) — and returns **one deterministic
decision**: approve / reject / route-to-agent, with a refund amount, routing
flags, and an AI recommendation.

This is an **extension of the existing system**, not a new build. It reuses the
service's money-safe spine — *the model observes, the application decides* — and
inherits the production/scale machinery already in place (stateless design,
Postgres audit, Redis dedup, GCS storage, idempotency, fail-fast config,
`/health`+`/ready`, structured logs, rate-limiting).

The existing `/scan`, `/verify-claim`, and `/evaluate-links` endpoints are
**unchanged** and out of scope.

## 2. Scope

One endpoint handling all requirement categories:

1. Wrong Product Received
2. Poor Quality
3. Damaged / Defective / Broken / Leakage / Torn / Crushed / Tampered
4. Expired / Near-Expiry (non-FNV 45-day rule; dairy 30%-shelf rule)
5. Stinking / Bad Smell
6. MRP Abuse / Mismatch (1P and 3P routing)
7. Quantity Mismatch

Product coverage: **FNV**, **Non-FNV**, **Dairy**. The caller supplies
`product_type` (resolved from the externally-maintained approved dairy list).

### Integration boundary (decided)

ImgRec stays a **stateless verifier**. The caller (Kaily/orchestrator) fetches
Shipment Details API + Kapture CRM data and passes it in the request body. ImgRec
does **not** call those APIs directly — this keeps the service fast, testable, and
decoupled from external-API uptime (the requirement's "MUST integrate with the
Shipment Details API" is met at the system level: the orchestrator integrates,
ImgRec consumes). The 99.9% Shipment-API uptime target is therefore owned
upstream.

## 3. Research basis (regulatory)

The 45-day / 30%-shelf rules are current Indian regulation. **FSSAI requires
e-commerce food sellers to deliver products with ≥30% shelf life OR ≥45 days
remaining** at delivery. The requirement encodes this correctly (45-day for
long-shelf non-FNV; 30%-shelf for short-shelf dairy). JioMart was flagged
non-compliant on best-before display in the June 2026 LocalCircles/FSSAI review,
so this carries regulatory + reputational urgency.

Sources: The Week (2026-06-18) quick-commerce expiry; Business Standard
(2026-06-23) FSSAI/DCA e-commerce expiry rules; FSSAI Labelling & Display
Regulations v8 (2025-09-09).

## 4. API contract

### Request

```jsonc
POST /api/v1/imgrecog/dispute
x-api-key: <KAILY_API_SECRET>
Content-Type: application/json
{
  "images": ["data:image/jpeg;base64,..."],   // 1..N customer photos (required, >=1)
  "dispute_category": "mrp_abuse",             // optional; null/omitted => classify from text
  "is_rebuttal": false,                        // true => customer disputing a prior rejection

  "ticket": {
    "title": "Charged more than printed price",
    "description": "...",                       // may be blank -> fallback chain
    "notes": "...",                             // Kapture notes, may be blank
    "disposition_code": "PRICE_DISPUTE"         // may be blank
  },

  "shipment": {
    "order_tracking_id": "JM-29384",
    "product_name": "Amul Gold Milk 500ml + free spoon",
    "product_type": "dairy",                     // fnv | non_fnv | dairy
    "mrp": 33.0,                                 // MRP per invoice/order
    "selling_price": 31.0,                       // per-unit charged
    "invoice_amount": 62.0,                      // total billed for the product line
    "quantity": 2,
    "seller_type": "1P"                          // 1P | 3P
  },

  "idempotency_key": "optional",
  "claim_id": "optional"
}
```

### Response

```jsonc
200 OK
{
  "success": true,
  "request_id": "dsp_1750...",
  "order_tracking_id": "JM-29384",

  "category": "mrp_abuse",
  "category_source": "provided",   // provided | description | notes | disposition

  "decision": "approve",           // approve | reject | agent
  "route": "auto",                 // auto | agent
  "agent_flags": [],               // counterfeit | rebuttal | high_value | insufficient_data |
                                   // internal_defect | low_confidence | missing_shipment_data | fraud_signal

  "refund": {
    "eligible": true,
    "amount": 4.0,
    "type": "price_difference",    // price_difference | full_selling_price | none
    "assign_to_mpt": false,        // true for 3P approved overcharge
    "seller_debit": false          // true for 3P approved overcharge
  },

  "recommendation": "Approve ₹4 price-difference refund (1P overcharge: printed MRP ₹31 < invoice MRP ₹33 × qty 2).",
  "confidence": 0.9,

  "observations": {
    "ocr": { "printed_mrp": 31.0, "printed_mrp_all": [49.0, 31.0], "expiry_date": null, "manufacture_date": null, "raw_text": "..." },
    "visual": { "damage": null, "quality": null, "counted_units": null, "detected_product": "Amul Gold Milk 500ml" },
    "scene": "A milk pouch held in hand",
    "ai_generated": { "is_ai_generated": false, "ai_probability": 0.04 }
  },

  "processed_at": "2026-06-25T10:30:00Z",
  "model_used": "gemini-2.5-flash"
}
```

### Errors
Same conventions as existing endpoints: `400/422` validation, `401` bad key,
`413` oversized image, `429` rate-limit (Retry-After), `503` overload/shed,
`504` Gemini timeout. Missing shipment fields required by the resolved category →
`decision: agent` + `missing_shipment_data` (never a crash).

## 5. Architecture

Chosen approach: **one new endpoint reusing the existing engine spine.**
(Considered and rejected: extending `/verify-claim` — bloats a working
fraud-scoring endpoint; a separate microservice — duplicates audit/auth/config.)

```
POST /api/v1/imgrecog/dispute  (auth · rate-limit · request-id · idempotency · audit)
  -> routers/dispute.py
       1. validate + size-check each image; downscale for model; compute pHash
       2. idempotency replay check (audit_service)  -> replay returns prior decision, no model call
       3. category_classifier -> category + category_source (fallback chain)
       4. concurrently:  dispute_service.analyze(...) [Gemini]   ||   fraud signals
                         (ai_image_detector, dedup_service, web_provenance)
       5. dispute_engine.decide(category, observations, shipment, signals)
       6. escalation gates (counterfeit / rebuttal / threshold / insufficient / fraud / low-conf)
       7. persist_decision (audit) ; safe downgrade to agent if audit write fails
       8. emit metrics (latency, outcome, category)
  -> DisputeResponse
```

### New files
- `app/routers/dispute.py` — endpoint orchestration (mirrors `verify.py`).
- `app/models/dispute_request.py` — `DisputeRequest`, `Ticket`, `Shipment`.
- `app/models/dispute_response.py` — `DisputeResponse`, `RefundResult`, `Observations`.
- `app/services/dispute_service.py` — multi-image Gemini observe call + JSON parse.
- `app/services/dispute_engine.py` — deterministic per-category decision + refund math + escalation.
- `app/services/category_classifier.py` — fallback chain.

### Reused (no behavior change)
`gemini_service`, `ocr_parser` (extended), `damage_analyzer` (extended),
`ai_image_detector`, `dedup_index`/`dedup_service`, `web_provenance`,
`audit_service`, `db/*`, `storage/object_store`, `middleware/*`, `utils/*`,
`config/settings` (+ new vars), `image_utils`.

### Extensions to reused modules
- `ocr_parser.py`: extract `printed_mrp` + `printed_mrp_all`; add dairy
  `shelf_left_pct(mfg, exp, today)`.
- `damage_analyzer.py`: add `tamper`, `broken_seal`, `resealed`,
  `missing_component` to the taxonomy.

## 6. Category resolution (requirement §3 fallback chain)

Strict order; record `category_source`:
1. `dispute_category` valid → `provided`.
2. classify from `ticket.description` → `description`.
3. from `ticket.notes` → `notes`.
4. map `ticket.disposition_code` → `disposition`
   (`WRONG_ITEM→wrong_product`, `QUALITY_ISSUE→poor_quality`, `DAMAGE→damaged`,
   `EXPIRY→expiry`, `PRICE_DISPUTE→mrp_abuse`).
5. none → `decision: agent` + `insufficient_data`.

## 7. Per-category decision logic (deterministic)

| Category | Approve when | Reject when | → Agent when |
|---|---|---|---|
| **wrong_product** | product in image ≠ ordered `product_name` | matches order | counterfeit mention |
| **poor_quality** | visual evidence (discoloration/wilting[FNV]/surface damage/defect) supports | looks normal | internal defect / warranty / performance |
| **damaged** | damage visually confirmed (seal/leak/tear/crush/tamper) | packaging intact | tamper ambiguous + high value |
| **expiry — non-FNV** | expiry ≤ 45 days from today | expiry > 45 days | expiry unreadable |
| **expiry — dairy** | shelf-left < 30% (needs MFG+EXP) | shelf-left ≥ 30% | MFG/EXP unreadable |
| **expiry — FNV** | → routed to **poor_quality** logic (variable shelf life) | | |
| **smell** | detailed description **and** image proxy (mold/discoloration) | insufficient evidence | — (§2.5 literal) |
| **mrp_abuse** | printed MRP `<` invoice MRP | printed MRP `≥` invoice MRP | MRP unreadable |
| **quantity_mismatch** | counted units `<` ordered `quantity` (confident) | counts match | low-confidence/occluded count |

**MRP comparison (resolves §2.6 ambiguity):** overcharge = MRP **printed on the
delivered pack** (OCR, post-strikethrough final value) is **strictly less than**
the MRP charged per invoice/order. Equality → reject. Prioritize **invoice
amount** over selling price when API values disagree.

## 8. MRP refund math (§2.6)

Approved `mrp_abuse`:
- **1P** → `(invoice_unit_price − printed_mrp) × quantity`; `type: price_difference`.
- **3P** → full `selling_price × quantity` (or `invoice_amount`);
  `type: full_selling_price`, `assign_to_mpt: true`, `seller_debit: true`.

Other approved categories → `type: full_selling_price` (or `invoice_amount`).

## 9. Escalation gates (override approve/reject → `agent`; recommendation always included)

1. Counterfeit (text/visual) → `counterfeit`.
2. Rebuttal (`is_rebuttal`) → `rebuttal`.
3. Refund ≥ `REFUND_AUTO_APPROVE_MAX` (default ₹500) → `high_value`.
4. No category resolvable → `insufficient_data`.
5. Internal defect / warranty / performance (poor_quality sub-case) → `internal_defect`.
6. Fraud signals (reused): AI-generated (advisory→agent; specialist→reject),
   cross-claim duplicate→reject, web-provenance public reuse→reject → `fraud_signal`.
7. Low-confidence visual read → `low_confidence`.
8. `DISPUTE_ASSIST_MODE=true` OR category not in `DISPUTE_AUTONOMOUS_CATEGORIES`
   → `route: agent` (recommend-only), regardless of decision. Enables shadow /
   progressive rollout without a redeploy.

---

## 10. Non-Functional Design

### 10.1 Accuracy (target ≥95% on complete data)

- **Determinism for money:** every refund/approve/reject is computed in
  `dispute_engine.py` from observations + shipment math — the model never decides.
  MRP, expiry, dairy-shelf, and quantity outcomes are arithmetic → accurate by
  construction given correct OCR.
- **Structured observations only:** one Gemini call returns strict JSON
  (response schema / JSON MIME, as `claim_service` does today). Malformed/empty →
  analysis failure → agent.
- **Confidence + contradiction guards:** low-confidence visual reads (e.g.
  quantity counts on occluded items) and internally-contradictory model output
  route to agent (mirrors `decision_engine` safeguards).
- **OCR robustness:** reuse `ocr_parser` Indian date handling; parse both
  strikethrough MRP values and use the final/lower one; multi-image OCR aggregation.
- **Reproducibility:** `DISPUTE_PROMPT_VERSION` recorded on every audit row, so a
  decision is reproducible against the exact prompt that produced it.
- **Eval harness (new, dev/CI):** a labelled fixture set of real JioMart dispute
  photos per category; a script computes per-category precision/recall and
  approve/reject confusion vs. ground truth. Thresholds (`*_THRESHOLD`,
  near-expiry days, shelf %) are tuned against this set, not guessed.
- **Progressive autonomy:** `DISPUTE_AUTONOMOUS_CATEGORIES` starts with the
  deterministic categories (mrp_abuse, expiry, wrong_product, damaged); subjective
  ones (poor_quality, smell, quantity_mismatch) stay assist-only until eval +
  shadow agreement clears ≥95%, then are added — no code change.

### 10.2 Speed (target p50 < 5s, p99 < 15s, hard cap < 30s)

- **One model call per dispute** (multi-image in a single request) — never
  per-category fan-out.
- **Image downscaling before model** (reuse `link_eval` 1280px max edge, q85) →
  smaller upload, faster inference, lower cost; original bytes kept for
  authenticity/provenance.
- **Concurrency:** Gemini call and fraud signals (ai-detector, dedup,
  web-provenance) run concurrently with `asyncio`, with the existing
  task-cancellation-on-timeout guard so a slow signal can't blow the budget.
- **Thinking budget disabled** on Gemini 2.5 (already done) to cut latency + JSON
  truncation.
- **Idempotent replay** returns the prior decision with **no model call**.
- **Hard timeout** `GEMINI_TIMEOUT_SECONDS` → 504; every external call is bounded.

### 10.3 Scalability (linear horizontal scale)

- **Stateless service** → scale by adding instances (Boltic 10→100). All state in
  Postgres (audit), Redis (dedup + rate-limit), GCS (images).
- **Global rate limiting (gap to close):** SlowAPI is currently in-memory per
  instance — back it with **Redis** (`storage_uri="redis://…"`) so the per-key
  limit is global across replicas. Token-bucket per `x-api-key`; `429` +
  `Retry-After`.
- **Load shedding + backpressure:** a bounded **concurrency semaphore** on Gemini
  calls per instance (respects Vertex quota), and a **circuit breaker** on Gemini
  so a model outage fast-fails to agent-routing instead of exhausting workers.
  Fast `503` under overload beats meltdown.
- **Async I/O end-to-end:** async FastAPI handler, async Gemini client, async
  SQLAlchemy with `pool_pre_ping` + a tuned pool size; no blocking calls on the
  hot path.
- **No N+1 / no chatty external calls:** shipment data arrives in the request
  (zero per-dispute external fetches from ImgRec).
- **Bounded everything:** image count per request capped, image size capped,
  retries capped (jittered, idempotent-only), audit/dedup windows bounded.

### 10.4 Production readiness

- **Inherited (no new work):** API-key auth (default-closed), fail-fast prod
  config guard, `/health` (liveness) + `/ready` (readiness — gate the LB),
  request-ID correlation, structured JSON logs, idempotency, durable Postgres
  audit, GCS image storage (India residency), safe-downgrade-to-agent on audit
  failure, no stack traces in responses.
- **Config & secrets:** new env vars below; secrets stay in Boltic's secret store
  (never committed). Prod config guard extended to validate the new money-affecting
  vars at boot.
- **Observability (new):** export the **four golden signals** (latency p50/p99,
  traffic, error rate, saturation = Gemini concurrency / DB pool) plus
  **per-category outcome counters** (approve/reject/agent) and **approve-rate
  gauge**. Alerts on: p99 latency > 15s, error rate > 1%, Gemini-timeout rate,
  escalation rate > 5% (the requirement bound), and **approve-rate drift** (fraud
  / model-regression early warning). Alert on symptoms, not CPU.
- **Migrations:** reversible Alembic **expand/contract** migration adding dispute
  columns to `claim_decisions` (or a sibling table): `category`,
  `category_source`, `decision`, `agent_flags`, `refund_*`, `dispute_prompt_version`.
  Old and new app versions both work during rollout.
- **Safe deploys:** one immutable artifact promoted dev→staging→prod; rollout via
  the progressive `DISPUTE_AUTONOMOUS_CATEGORIES` / `DISPUTE_ASSIST_MODE` flags
  (assist → per-category autonomy), giving a canary-like ramp with instant
  rollback by flag.
- **Resilience:** timeouts + jittered retries (idempotent ops only) + circuit
  breaker on Gemini; graceful shutdown (drain in-flight on SIGTERM — uvicorn).
- **Security & PII:** validate/sanitise all input at the boundary; ticket text may
  contain PII — **do not log raw ticket text** (log lengths/flags only); audit row
  stores it behind existing access controls with `image_retention_days` cleanup;
  CVE-scan dependencies in CI; reuse the SSRF-safe image handling.
- **SLOs:** 99.9% availability, p99 < 15s, ≥95% accuracy per *autonomous*
  category; error budget governs feature-vs-reliability work.

## 11. Configuration (new env vars)

| Var | Default | Meaning |
|---|---|---|
| `REFUND_AUTO_APPROVE_MAX` | `500` | Refund ≥ this (₹) → agent approval |
| `DISPUTE_ASSIST_MODE` | `false` | `true` = recommend-only (no auto-act) |
| `DISPUTE_AUTONOMOUS_CATEGORIES` | `mrp_abuse,expiry,wrong_product,damaged` | Categories allowed to auto-act |
| `DAIRY_MIN_SHELF_PCT` | `30` | Dairy min remaining shelf % |
| `NON_FNV_NEAR_EXPIRY_DAYS` | `45` | Non-FNV near-expiry threshold (days) |
| `DISPUTE_MAX_IMAGES` | `5` | Max images per dispute |
| `GEMINI_MAX_CONCURRENCY` | `8` | Per-instance semaphore on model calls |
| `DISPUTE_PROMPT_VERSION` | `dispute-v1` | Recorded on each audit row |

Existing prod guard (auth, DB, object store, Redis) already covers this endpoint.

## 12. Error handling

- Gemini timeout → 504; malformed/empty JSON → agent.
- Invalid/oversized image → 400/413; too many images → 422.
- Multi-image: analyse all in one call; aggregate **worst-case** evidence.
- Missing shipment field for resolved category → agent + `missing_shipment_data`.
- Audit write failure → downgrade to agent (existing money-safe behavior).

## 13. Testing

- `tests/test_dispute_engine.py` — every category's approve/reject/agent matrix;
  MRP 1P/3P math + strikethrough; dairy 30% boundary; 45-day boundary; FNV→quality
  routing; escalation gates; refund-ceiling; assist-mode + autonomous-category
  gating.
- `tests/test_category_classifier.py` — full fallback chain + INSUFFICIENT_DATA.
- `tests/test_dispute.py` — endpoint: auth, validation, idempotency replay,
  multi-image, Gemini timeout, concurrency/circuit-breaker degradation, Gemini
  mocked.
- `tests/eval/` — labelled-fixture accuracy harness (per-category precision/recall),
  run in CI as a non-gating report initially.
- Follow existing pattern: mock Gemini at the service boundary; reset in-memory
  stores per test.

## 14. Implementation phasing (single spec, ordered build)

1. **Deterministic core:** models + endpoint + `dispute_engine` for mrp_abuse,
   expiry/dairy, wrong_product, damaged; category_classifier; audit + migration;
   idempotency; new request/response contract.
2. **Remaining categories:** quantity_mismatch, poor_quality, smell; ocr_parser +
   damage_analyzer extensions; worst-case multi-image aggregation.
3. **NFR hardening + rollout:** Redis-backed rate limit, Gemini semaphore +
   circuit breaker, metrics + alerts, eval harness, assist/autonomous-category
   flags, README + code-guide updates, `.env.example` vars.

## 15. Out of scope

- ImgRec calling Shipment Details API / Kapture directly (orchestrator owns it).
- 1P/3P ticket execution & MPT seller-debit *workflow* (ImgRec emits flags;
  Kaily/Kapture executes).
- Maintaining the approved dairy product list (external).
- Changes to `/scan`, `/verify-claim`, `/evaluate-links`.
```
