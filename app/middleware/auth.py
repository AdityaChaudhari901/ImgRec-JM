from typing import Optional

from fastapi import Header, HTTPException, status

from app.config.settings import settings

_LOCAL_DEMO_API_KEY = "missing-kaily-secret"


def _is_local_development() -> bool:
    return settings.environment.strip().lower() == "development"


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Validate the shared Kaily secret passed via the `x-api-key` header.

    Both a missing header and a present-but-wrong key yield 401 (per the API
    contract), rather than letting a missing header fall through to a 422.
    """
    if x_api_key == settings.kaily_api_secret:
        return

    # The bundled static demo cannot read `.env`. In local development only, let
    # it use the placeholder key already embedded in demo/index.html. Do not
    # allow this in staging/production where the real shared secret is required.
    if _is_local_development() and x_api_key == _LOCAL_DEMO_API_KEY:
        return

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
