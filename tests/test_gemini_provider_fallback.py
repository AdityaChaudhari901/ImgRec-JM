from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.config.settings import settings
from app.services import gemini_service


class _FakeModels:
    def __init__(self, provider: str, calls: list[str]):
        self.provider = provider
        self.calls = calls

    async def generate_content(self, **kwargs):
        self.calls.append(self.provider)
        if self.provider == "api_key":
            raise genai_errors.APIError(
                429,
                {
                    "error": {
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "Your prepayment credits are depleted.",
                    }
                },
            )
        return SimpleNamespace(text="ok")


class _FakeClient:
    def __init__(self, provider: str, calls: list[str]):
        self.aio = SimpleNamespace(models=_FakeModels(provider, calls))


@pytest.mark.asyncio
async def test_ai_studio_quota_error_retries_with_vertex(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(settings, "use_vertex", False)
    monkeypatch.setattr(settings, "gemini_vertex_fallback_enabled", True)
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    monkeypatch.setattr(
        gemini_service,
        "get_client",
        lambda provider=None: _FakeClient(provider or "api_key", calls),
    )

    response = await gemini_service.generate_content_with_fallback(
        model="gemini-test",
        contents=["prompt"],
        config=gemini_service.build_dispute_generation_config(),
    )

    assert response.text == "ok"
    assert calls == ["api_key", "vertex"]


@pytest.mark.asyncio
async def test_ai_studio_quota_error_does_not_retry_when_disabled(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(settings, "use_vertex", False)
    monkeypatch.setattr(settings, "gemini_vertex_fallback_enabled", False)
    monkeypatch.setattr(settings, "vertex_project_id", "test-project")
    monkeypatch.setattr(
        gemini_service,
        "get_client",
        lambda provider=None: _FakeClient(provider or "api_key", calls),
    )

    with pytest.raises(genai_errors.APIError):
        await gemini_service.generate_content_with_fallback(
            model="gemini-test",
            contents=["prompt"],
            config=gemini_service.build_dispute_generation_config(),
        )

    assert calls == ["api_key"]
