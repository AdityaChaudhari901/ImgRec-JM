import asyncio
from unittest.mock import patch

from eval_harness.dataset import EvalCase
from eval_harness.metrics import CasePrediction, compute_metrics
from eval_harness.report import format_report
from eval_harness.run import main
from eval_harness.runner import run_case_e2e


def test_format_report_contains_headline_numbers():
    preds = [
        CasePrediction("a", "approve", "approve", "mrp_abuse", "mrp_abuse"),
        CasePrediction("b", "reject", "approve", "expiry", "expiry"),
    ]
    text = format_report(compute_metrics(preds))
    assert "Decision accuracy" in text
    assert "Confusion" in text
    assert "b: expected reject, got approve" in text


def test_cli_engine_mode_passes_threshold(capsys):
    # The seed set is 100% in engine mode, so a 0.95 gate must pass (exit 0).
    code = main(["--mode", "engine", "--threshold", "0.95"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Decision accuracy: **100.00%**" in out


def test_cli_threshold_failure_exits_nonzero(tmp_path):
    # A single case the engine will reject but we label approve -> 0% -> exit 1.
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(
        '{"id":"x","category":"mrp_abuse","shipment":{"order_tracking_id":"1",'
        '"product_name":"Oil","product_type":"non_fnv","mrp":100,"selling_price":100,'
        '"invoice_amount":100,"quantity":1,"seller_type":"1P"},'
        '"observations":{"ocr":{"printed_mrp_values":[100]}},'
        '"expected":{"decision":"approve","category":"mrp_abuse"}}\n'
    )
    assert main(["--manifest", str(manifest), "--threshold", "0.95"]) == 1


def test_cli_model_override_sets_gemini_model():
    from app.config.settings import settings
    original = settings.gemini_model
    try:
        main(["--mode", "engine", "--model", "gemini-2.5-pro", "--threshold", "0"])
        assert settings.gemini_model == "gemini-2.5-pro"
    finally:
        settings.gemini_model = original


def test_e2e_runner_wires_gemini_observations():
    case = EvalCase(
        id="e1", category="mrp_abuse", expected_decision="approve",
        expected_category="mrp_abuse", images=["data:image/jpeg;base64,AAAA"],
        shipment={"order_tracking_id": "JM-1", "product_name": "Oil", "product_type": "non_fnv",
                  "mrp": 100, "selling_price": 100, "invoice_amount": 100,
                  "quantity": 1, "seller_type": "1P"},
    )

    async def fake_analyze(*a, **k):
        return {"ocr": {"printed_mrp_values": [90]}, "ai_generated": {"ai_probability": 0.0}}

    with patch("app.services.dispute_service.analyze_dispute", side_effect=fake_analyze):
        pred = asyncio.get_event_loop().run_until_complete(run_case_e2e(case))
    assert pred.predicted_decision == "approve"
