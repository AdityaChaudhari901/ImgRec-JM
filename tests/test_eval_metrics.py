from eval_harness.metrics import CasePrediction, compute_metrics


def _p(eid, ed, pd, ec=None, pc=None):
    return CasePrediction(id=eid, expected_decision=ed, predicted_decision=pd,
                          expected_category=ec, predicted_category=pc)


def test_perfect_accuracy():
    preds = [
        _p("a", "approve", "approve", "mrp_abuse", "mrp_abuse"),
        _p("b", "reject", "reject", "expiry", "expiry"),
    ]
    m = compute_metrics(preds)
    assert m.total == 2
    assert m.decision_accuracy == 1.0
    assert m.category_accuracy == 1.0
    assert m.approve_precision == 1.0
    assert m.approve_recall == 1.0


def test_mixed_accuracy_and_confusion():
    preds = [
        _p("a", "approve", "approve"),   # correct approve
        _p("b", "approve", "reject"),    # missed approve (false reject)
        _p("c", "reject", "approve"),    # false approve
        _p("d", "agent", "agent"),       # correct agent
    ]
    m = compute_metrics(preds)
    assert m.total == 4
    assert m.decision_accuracy == 0.5  # 2/4 correct
    # approve precision: predicted approve = {a, c}; correct = {a} -> 1/2
    assert m.approve_precision == 0.5
    # approve recall: expected approve = {a, b}; caught = {a} -> 1/2
    assert m.approve_recall == 0.5
    assert m.decision_confusion["approve"]["reject"] == 1
    assert m.decision_confusion["reject"]["approve"] == 1


def test_per_category_breakdown():
    preds = [
        _p("a", "approve", "approve", "mrp_abuse", "mrp_abuse"),
        _p("b", "reject", "approve", "mrp_abuse", "mrp_abuse"),
        _p("c", "approve", "approve", "expiry", "expiry"),
    ]
    m = compute_metrics(preds)
    assert m.per_category["mrp_abuse"]["n"] == 2
    assert m.per_category["mrp_abuse"]["decision_accuracy"] == 0.5
    assert m.per_category["expiry"]["decision_accuracy"] == 1.0


def test_empty_is_safe():
    m = compute_metrics([])
    assert m.total == 0
    assert m.decision_accuracy == 0.0
