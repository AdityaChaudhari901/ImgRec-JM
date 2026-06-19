from app.config.settings import settings


def test_web_provenance_defaults():
    assert settings.web_provenance_enabled is True
    assert settings.web_match_hard_min_domains == 2
    assert settings.web_match_soft_penalty == 0.15
    assert settings.web_match_penalty_cap == 3
    assert settings.vision_timeout_seconds == 8
