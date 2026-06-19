from app.services.decision_engine import (
    build_response,
    determine_action,
    generate_request_id,
)


def test_expired_triggers_high_priority_refund():
    action = determine_action("expired", severity=None, days_since_expiry=28)
    assert action.type == "initiate_refund"
    assert action.refund_eligible is True
    assert action.priority == "high"
    assert "28" in action.message


def test_severe_damage_triggers_refund():
    action = determine_action("damaged", severity="severe")
    assert action.type == "initiate_refund"
    assert action.refund_eligible is True
    assert action.priority == "high"


def test_moderate_damage_triggers_exchange():
    action = determine_action("damaged", severity="moderate")
    assert action.type == "initiate_exchange"
    assert action.refund_eligible is False
    assert action.priority == "medium"


def test_minor_damage_triggers_exchange_low():
    action = determine_action("damaged", severity="minor")
    assert action.type == "initiate_exchange"
    assert action.refund_eligible is False
    assert action.priority == "low"


def test_valid_triggers_no_action():
    action = determine_action("valid", severity=None)
    assert action.type == "no_action"
    assert action.refund_eligible is False
    assert action.priority == "low"


def test_unclear_triggers_no_action():
    action = determine_action("unclear", severity=None)
    assert action.type == "no_action"
    assert action.refund_eligible is False


def test_model_message_is_preferred_when_present():
    action = determine_action(
        "expired", severity=None, days_since_expiry=5, model_message="Custom note."
    )
    assert action.message == "Custom note."


def test_no_action_ignores_refund_model_message():
    action = determine_action(
        "unclear",
        severity="severe",
        model_message="We will initiate a full refund for this item.",
    )
    assert action.type == "no_action"
    assert action.refund_eligible is False
    assert "refund" not in action.message.lower()


def test_request_id_format():
    rid = generate_request_id()
    assert rid.startswith("req_")
    assert len(rid.split("_")) == 3


def test_build_response_full_object():
    gemini = {
        "status": "expired",
        "confidence": 0.96,
        "ocr": {
            "manufacture_date": "JAN 2024",
            "expiry_date": "2023-05-20",
            "batch_no": "B2401K",
            "raw_text": "MFG JAN 2024 EXP MAY 2023",
        },
        "damage": {"detected": False, "type": None, "severity": None, "description": None},
        "ai_generated": {"ai_probability": 0.02, "signals": []},
        "action": {"type": "initiate_refund", "message": "Expired.", "refund_eligible": True, "priority": "high"},
    }
    resp = build_response(gemini, order_id="JM-29384", user_id="u_kaily_123")
    assert resp.success is True
    assert resp.order_id == "JM-29384"
    assert resp.status == "expired"
    assert resp.ocr.expiry_date == "2023-05-20"
    assert resp.ocr.manufacture_date == "2024-01-31"  # normalised from "JAN 2024"
    assert resp.ocr.days_since_expiry and resp.ocr.days_since_expiry > 0
    assert resp.action.type == "initiate_refund"
    assert resp.action.refund_eligible is True


def test_build_response_rejects_invalid_status():
    gemini = {"status": "bogus", "confidence": 0.1, "ocr": {}, "damage": {}, "action": {}}
    resp = build_response(gemini, order_id="JM-1", user_id="u_1")
    assert resp.status == "unclear"
    assert resp.action.type == "no_action"


def test_unclear_with_severe_damage_and_refund_message_routes_to_manual_review():
    gemini = {
        "status": "unclear",
        "confidence": 0.7,
        "ocr": {"raw_text": "warped milk label text"},
        "damage": {
            "detected": True,
            "type": "leakage",
            "severity": "severe",
            "description": "Milk pouch is torn and leaking.",
        },
        "action": {
            "type": "no_action",
            "message": "We will initiate a full refund for this item.",
            "refund_eligible": False,
            "priority": "low",
        },
    }
    resp = build_response(gemini, order_id="JM-1", user_id="u_1")

    assert resp.status == "unclear"
    assert resp.damage.detected is False
    assert resp.damage.type is None
    assert resp.action.type == "no_action"
    assert resp.action.refund_eligible is False
    assert resp.action.priority == "high"
    assert "manual review" in resp.action.message
    assert "refund for this item" not in resp.action.message


def test_damaged_refund_without_authenticity_assessment_routes_to_manual_review():
    gemini = {
        "status": "damaged",
        "confidence": 0.9,
        "ocr": {"raw_text": "warped milk label text"},
        "damage": {
            "detected": True,
            "type": "leakage",
            "severity": "severe",
            "description": "Milk pouch is torn and leaking.",
        },
        "action": {
            "type": "initiate_refund",
            "message": "A refund has been initiated for this product.",
            "refund_eligible": True,
            "priority": "high",
        },
    }
    resp = build_response(gemini, order_id="JM-1", user_id="u_1")

    assert resp.status == "unclear"
    assert resp.damage.detected is False
    assert resp.action.type == "no_action"
    assert resp.action.refund_eligible is False
    assert resp.action.priority == "high"
    assert "authenticity was not assessed" in resp.action.message
