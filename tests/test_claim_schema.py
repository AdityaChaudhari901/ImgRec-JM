from app.services.gemini_service import build_claim_generation_config


def test_claim_config_has_json_mime_and_schema():
    cfg = build_claim_generation_config()
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_schema is not None
