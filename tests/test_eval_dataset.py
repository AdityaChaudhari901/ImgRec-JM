import json

from eval_harness.dataset import EvalCase, load_manifest


def test_load_manifest_parses_jsonl(tmp_path):
    rows = [
        {"id": "c1", "category": "mrp_abuse", "shipment": {"order_tracking_id": "JM-1",
         "product_name": "Oil", "product_type": "non_fnv", "mrp": 100, "selling_price": 100,
         "invoice_amount": 100, "quantity": 1, "seller_type": "1P"},
         "observations": {"ocr": {"printed_mrp_values": [90]}},
         "expected": {"decision": "approve", "category": "mrp_abuse"}},
        {"id": "c2", "category": None, "ticket": {"description": "expired"},
         "shipment": {"order_tracking_id": "JM-2", "product_name": "Biscuit",
         "product_type": "non_fnv", "quantity": 1, "seller_type": "1P"},
         "observations": {"ocr": {"expiry_date": "2026-07-01"}},
         "today": "2026-06-25",
         "expected": {"decision": "approve", "category": "expiry"}},
    ]
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    cases = load_manifest(p)
    assert len(cases) == 2
    assert isinstance(cases[0], EvalCase)
    assert cases[0].expected_decision == "approve"
    assert cases[0].category == "mrp_abuse"
    assert cases[1].category is None
    assert cases[1].today == "2026-06-25"


def test_load_manifest_skips_blank_lines(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('\n  \n{"id":"x","shipment":{"order_tracking_id":"1","product_name":"a",'
                 '"product_type":"non_fnv","quantity":1,"seller_type":"1P"},'
                 '"observations":{},"expected":{"decision":"agent"}}\n')
    cases = load_manifest(p)
    assert len(cases) == 1
    assert cases[0].expected_category is None
