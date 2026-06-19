from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ScanRequest(BaseModel):
    image_base64: str = Field(..., description="Full data URI or raw base64 string")
    order_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    scan_type: Literal["auto", "ocr", "damage"] = "auto"
    # Idempotency: an explicit key (or claim_id) makes retries return the first
    # decision verbatim. If both are absent, a key is derived from
    # (order_id, user_id, image phash). Optional — keeps the contract backward-compatible.
    idempotency_key: Optional[str] = Field(default=None, max_length=200)
    claim_id: Optional[str] = Field(default=None, max_length=200)

    @field_validator("image_base64")
    @classmethod
    def validate_image(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("image_base64 must be a valid data URI or raw base64")
        return v
