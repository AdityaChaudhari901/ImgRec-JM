import json

from google.oauth2 import service_account

from app.config.settings import settings
from app.utils import gcp_auth


def test_returns_none_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "google_credentials_json", "")
    assert gcp_auth.google_sa_credentials() is None


def test_whitespace_only_is_treated_as_unset(monkeypatch):
    monkeypatch.setattr(settings, "google_credentials_json", "  \n  ")
    assert gcp_auth.google_sa_credentials() is None


def test_builds_credentials_from_inline_json(monkeypatch):
    info = {"type": "service_account", "project_id": "p", "client_email": "sa@p.iam"}
    monkeypatch.setattr(settings, "google_credentials_json", json.dumps(info))

    captured = {}
    sentinel = object()

    def fake_from_info(parsed, scopes):
        captured["parsed"] = parsed
        captured["scopes"] = scopes
        return sentinel

    monkeypatch.setattr(
        service_account.Credentials, "from_service_account_info", fake_from_info
    )

    result = gcp_auth.google_sa_credentials()

    assert result is sentinel
    assert captured["parsed"] == info
    assert captured["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
