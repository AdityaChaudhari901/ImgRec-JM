from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ProductType = Literal["fnv", "non_fnv", "dairy"]
SellerType = Literal["1P", "3P"]
DisputeCategory = Literal[
    "wrong_product", "poor_quality", "damaged", "expiry",
    "smell", "mrp_abuse", "quantity_mismatch",
]


class Ticket(BaseModel):
    title: str = Field(default="", max_length=500)
    description: str = Field(default="", max_length=4000)
    notes: str = Field(default="", max_length=4000)
    disposition_code: str = Field(default="", max_length=100)


class Shipment(BaseModel):
    order_tracking_id: str = Field(..., min_length=1)
    product_name: str = Field(..., min_length=1, max_length=500)
    product_type: ProductType
    mrp: Optional[float] = Field(default=None, ge=0)
    selling_price: Optional[float] = Field(default=None, ge=0)
    invoice_amount: Optional[float] = Field(default=None, ge=0)
    quantity: int = Field(default=1, ge=1)
    seller_type: SellerType = "1P"


class DisputeRequest(BaseModel):
    """A grocery dispute to resolve from images + ticket text + shipment data."""

    images: List[str] = Field(..., min_length=1)
    dispute_category: Optional[DisputeCategory] = None
    is_rebuttal: bool = False
    ticket: Ticket = Field(default_factory=Ticket)
    shipment: Shipment
    idempotency_key: Optional[str] = Field(default=None, max_length=200)
    claim_id: Optional[str] = Field(default=None, max_length=200)

    @field_validator("images")
    @classmethod
    def _non_empty_images(cls, v: List[str]) -> List[str]:
        cleaned = [s for s in v if s and s.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty image is required")
        return cleaned
