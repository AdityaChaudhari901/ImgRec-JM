import pytest
from fastapi import HTTPException

from app.config.settings import settings
from app.middleware.auth import verify_api_key


@pytest.mark.asyncio
async def test_configured_api_key_is_accepted(monkeypatch):
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "kaily_api_secret", "real-local-secret")

    await verify_api_key("real-local-secret")


@pytest.mark.asyncio
async def test_demo_api_key_is_accepted_only_in_development(monkeypatch):
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "kaily_api_secret", "real-local-secret")

    await verify_api_key("missing-kaily-secret")


@pytest.mark.asyncio
async def test_demo_api_key_is_rejected_outside_development(monkeypatch):
    monkeypatch.setattr(settings, "environment", "staging")
    monkeypatch.setattr(settings, "kaily_api_secret", "real-staging-secret")

    with pytest.raises(HTTPException) as exc:
        await verify_api_key("missing-kaily-secret")

    assert exc.value.status_code == 401
