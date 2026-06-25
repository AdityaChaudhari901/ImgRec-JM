from eval_harness.dataset import EvalCase
from eval_harness.runner import run_case, run_dataset


def _case(**o):
    base = dict(
        id="t", shipment={"order_tracking_id": "JM-1", "product_name": "Oil",
                          "product_type": "non_fnv", "mrp": 100, "selling_price": 100,
                          "invoice_amount": 100, "quantity": 1, "seller_type": "1P"},
        expected_decision="approve", category="mrp_abuse",
        observations={"ocr": {"printed_mrp_values": [90]}},
    )
    base.update(o)
    return EvalCase(**base)


def test_run_case_mrp_approve():
    pred = run_case(_case())
    assert pred.predicted_decision == "approve"
    assert pred.predicted_category == "mrp_abuse"


def test_run_case_classifies_when_category_none():
    c = _case(category=None, ticket={"description": "I was overcharged"})
    pred = run_case(c)
    assert pred.predicted_category == "mrp_abuse"
    assert pred.predicted_decision == "approve"


def test_run_case_expiry_uses_case_today():
    c = _case(
        category="expiry",
        observations={"ocr": {"expiry_date": "2026-07-01"}},
        today="2026-06-25",
        expected_decision="approve",
    )
    assert run_case(c).predicted_decision == "approve"  # 6 days <= 45


def test_run_case_insufficient_data_agent():
    c = _case(category=None, ticket={}, observations={}, expected_decision="agent")
    pred = run_case(c)
    assert pred.predicted_decision == "agent"
    assert pred.predicted_category is None


def test_run_dataset_returns_predictions():
    preds = run_dataset([_case(), _case(id="t2")])
    assert len(preds) == 2
