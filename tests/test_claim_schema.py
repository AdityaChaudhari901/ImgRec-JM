from app.services.gemini_service import build_claim_generation_config


def test_claim_config_has_json_mime_and_schema():
    cfg = build_claim_generation_config()
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_schema is not None
    schema = cfg.response_schema
    assert set(schema["properties"].keys()) == {
        "recognition",
        "ai_generated",
        "image_comment_alignment",
        "product_match",
        "other_flags",
        "summary",
    }
