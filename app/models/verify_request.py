from typing import Optional

from pydantic import BaseModel, Field, field_validator


class VerifyClaimRequest(BaseModel):
    """A customer's damage/destroyed-product claim to be authenticity-scored."""

    image_base64: str = Field(..., description="Full data URI or raw base64 string")
    user_comment: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="What issue the customer says they are facing",
    )
    claimed_product: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Product the ticket is about (from the order/ticket context)",
    )
    order_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    # Optional catalog/reference image for a stronger product-match check.
    reference_image_base64: Optional[str] = Field(
        default=None, description="Optional reference product image (data URI or base64)"
    )
    # Idempotency (see ScanRequest): explicit key/claim_id, else derived from
    # (order_id, user_id, image phash). Optional — backward-compatible.
    idempotency_key: Optional[str] = Field(default=None, max_length=200)
    claim_id: Optional[str] = Field(default=None, max_length=200)

    @field_validator("image_base64")
    @classmethod
    def validate_image(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("image_base64 must be a valid data URI or raw base64")
        return v
