from app.config.settings import settings


def test_dispute_defaults():
    assert settings.refund_auto_approve_max == 500
    assert settings.dispute_assist_mode is False
    assert "mrp_abuse" in settings.dispute_autonomous_categories
    assert settings.dairy_min_shelf_pct == 30
    assert settings.non_fnv_near_expiry_days == 45
    assert settings.dispute_max_images == 5
    assert settings.dispute_prompt_version == "dispute-v1"
