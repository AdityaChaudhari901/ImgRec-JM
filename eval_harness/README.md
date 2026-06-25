# Dispute Accuracy Eval Harness

Measures how accurately the `/dispute` system reaches the right verdict. It scores
the **real** production classifier + decision engine (not a copy), and reports
per-category accuracy, an approve/reject/agent confusion matrix, and approve
precision/recall (a false approve loses money; a false reject is only a CX cost —
they're tracked separately).

## Two modes — two different accuracies

| Mode | What it tests | Needs | Cost |
|---|---|---|---|
| **engine** (default) | The deterministic decision logic: given observations, is the verdict right? | nothing | free, offline |
| **e2e** | The whole pipeline incl. **Gemini reading a real photo** | real images + provider creds | model calls |

> **Why both matter.** Total accuracy = `model_accuracy × engine_accuracy`. Engine
> mode proves the math is right (the seed set scores 100%, enforced in CI). e2e
> mode is the only thing that tells you whether **Gemini reads real customer
> photos** correctly — the actual unknown, and what the ≥95% target hinges on.

## Run it

```bash
# Engine mode — business logic, no API cost (also runs in CI as a golden test):
venv/bin/python -m eval_harness.run --threshold 0.95

# End-to-end — real Gemini over real images:
venv/bin/python -m eval_harness.run --mode e2e --manifest eval_harness/data/real.jsonl --threshold 0.95

# Compare models on the SAME labelled set (this answers "should we switch models?"):
venv/bin/python -m eval_harness.run --mode e2e --manifest real.jsonl --model gemini-2.5-flash
venv/bin/python -m eval_harness.run --mode e2e --manifest real.jsonl --model gemini-2.5-pro
```

Exit code is non-zero when decision accuracy is below `--threshold`, so it can gate
a CI job once a real labelled set exists.

## Collecting labelled data (the part only JioMart can do)

Every dispute a human agent has already resolved is a labelled example. From
Kapture / the audit store, for each resolved dispute capture:

1. **The customer image URL(s)** → `image_urls`.
2. **The agent's final decision** (approve / reject / sent-to-specialist) → `expected.decision`.
3. **The category** → `category` (and `expected.category`).
4. **The shipment fields** the category needs (MRP/selling/invoice/qty for
   `mrp_abuse`; `product_type` for expiry; etc.) → `shipment`.
5. For expiry cases, the **delivery date** the rule should evaluate against → `today`.

Aim for ~100–300 rows spread across the categories. Copy
`eval_harness/data/e2e_template.jsonl` and replace the `EXAMPLE-*` rows with real
data. Keep it out of version control if the URLs are sensitive.

## Manifest format (one JSON object per line)

```jsonc
{
  "id": "unique-id",
  "category": "mrp_abuse",          // or null to also test classification
  "is_rebuttal": false,
  "today": "2026-06-25",            // optional; pins "today" for expiry cases
  "image_urls": ["https://..."],    // e2e mode  (or "observations": {...} for engine mode)
  "ticket": { "description": "charged more than printed" },
  "shipment": { "order_tracking_id": "JM-1", "product_name": "Oil 1L",
                "product_type": "non_fnv", "mrp": 199, "selling_price": 199,
                "invoice_amount": 199, "quantity": 1, "seller_type": "1P" },
  "expected": { "decision": "approve", "category": "mrp_abuse" }   // ground truth
}
```

## Reading the result

- **Decision accuracy** — the headline. Per-category breakdown tells you *which*
  categories are safe to make autonomous.
- **Confusion matrix** — watch the `reject→approve` and `agent→approve` cells:
  those are false approvals (money lost).
- **Graduate a category to autonomous** (`DISPUTE_AUTONOMOUS_CATEGORIES`) only once
  it clears your bar (e.g. ≥95%) on real data. Keep weak ones (smell, poor-quality)
  assisted longer.
