from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """A grocery dispute to resolve from images + ticket text + shipment data.

    Images may be supplied as base64 (`images`) and/or public URLs (`image_urls`)
    — at least one image, by either route, is required. `shipment` is optional:
    when absent, categories that need it (MRP, quantity) route to an agent, while
    image-only categories (damaged, expiry, quality, smell, wrong-product) are
    still decided.
    """

    images: Optional[List[str]] = None
    image_urls: Optional[List[str]] = None
    dispute_category: Optional[DisputeCategory] = None
    is_rebuttal: bool = False
    ticket: Ticket = Field(default_factory=Ticket)
    shipment: Optional[Shipment] = None
    idempotency_key: Optional[str] = Field(default=None, max_length=200)
    claim_id: Optional[str] = Field(default=None, max_length=200)

    @field_validator("images", "image_urls")
    @classmethod
    def _drop_blanks(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        return [s for s in v if s and s.strip()]

    @model_validator(mode="after")
    def _require_at_least_one_image(self):
        if not (self.images or self.image_urls):
            raise ValueError("at least one image is required (images or image_urls)")
        return self
