"""Shared Google Cloud credential loading.

Serverless runtimes (e.g. Boltic) can't reliably mount a service-account key
file, so we accept the key JSON inline via the GOOGLE_CREDENTIALS_JSON secret.
Both the Vertex AI (Gemini) client and the Cloud Vision client use this so they
authenticate the same way. When the secret is unset, returning None lets
google-auth fall back to Application Default Credentials (a mounted
GOOGLE_APPLICATION_CREDENTIALS file or the workload-identity metadata server).
"""

import json

from app.config.settings import settings

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def google_sa_credentials():
    """Service-account credentials from GOOGLE_CREDENTIALS_JSON, or None for ADC."""
    raw = settings.google_credentials_json.strip()
    if not raw:
        return None
    from google.oauth2 import service_account

    return service_account.Credentials.from_service_account_info(
        json.loads(raw), scopes=_SCOPES
    )
