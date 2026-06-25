import pytest

from app.config.settings import Settings


def _prod(**over):
    base = dict(environment="production", kaily_api_secret="real", google_api_key="real",
                database_url="postgresql+asyncpg://x", object_store_provider="gcs",
                gcs_bucket="b", redis_url="redis://x")
    base.update(over)
    return base


def test_negative_refund_ceiling_rejected():
    with pytest.raises(ValueError):
        Settings(**_prod(refund_auto_approve_max=-1))


def test_bad_shelf_pct_rejected():
    with pytest.raises(ValueError):
        Settings(**_prod(dairy_min_shelf_pct=150))


def test_valid_prod_config_ok():
    Settings(**_prod())  # should not raise
