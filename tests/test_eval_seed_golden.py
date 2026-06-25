"""Golden regression: the deterministic engine must score 100% on the seed set.

This is the business-logic accuracy proof. Each seed case is a labelled boundary
(45-day edge, dairy 30% edge, MRP equality, high-value ceiling, the escalation
gates, classification fallback). A drop below 100% means a decision rule changed
behavior — a real regression, not noise.
"""

from eval_harness.dataset import SEED_MANIFEST, load_manifest
from eval_harness.metrics import compute_metrics
from eval_harness.runner import run_dataset


def test_seed_manifest_is_substantial():
    cases = load_manifest(SEED_MANIFEST)
    assert len(cases) >= 20
    # Every category is represented at least once.
    cats = {c.expected_category for c in cases if c.expected_category}
    assert {"mrp_abuse", "expiry", "wrong_product", "damaged",
            "poor_quality", "smell", "quantity_mismatch"} <= cats


def test_engine_scores_100_percent_on_seed():
    cases = load_manifest(SEED_MANIFEST)
    metrics = compute_metrics(run_dataset(cases))
    assert metrics.decision_accuracy == 1.0, f"mismatches: {metrics.mismatches}"
    assert metrics.category_accuracy == 1.0


def test_no_false_approvals_on_seed():
    # The money-critical guarantee: nothing the seed labels reject/agent is
    # predicted approve.
    cases = load_manifest(SEED_MANIFEST)
    metrics = compute_metrics(run_dataset(cases))
    false_approve = (
        metrics.decision_confusion["reject"]["approve"]
        + metrics.decision_confusion["agent"]["approve"]
    )
    assert false_approve == 0
