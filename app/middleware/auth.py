from typing import Optional

from fastapi import Header, HTTPException, status

from app.config.settings import settings


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Validate the shared Kaily secret passed via the `x-api-key` header.

    Both a missing header and a present-but-wrong key yield 401 (per the API
    contract), rather than letting a missing header fall through to a 422.
    """
    if not x_api_key or x_api_key != settings.kaily_api_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
