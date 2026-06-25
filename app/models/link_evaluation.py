from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class LinkedImageEvaluationRequest(BaseModel):
    """URL-first product claim evaluation request.

    `user_image_url` is the customer-provided evidence image.
    `product_image_url` is the official/catalog/reference product image.
    `query` is the customer's claim, such as "expired product" or
    "damaged product".
    """

    user_image_url: str = Field(..., min_length=8, max_length=4096)
    product_image_url: str = Field(..., min_length=8, max_length=4096)
    query: str = Field(..., min_length=1, max_length=500)

    @field_validator("user_image_url", "product_image_url", "query")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class EvaluationCheck(BaseModel):
    score: int = Field(..., ge=0, le=100)
    verdict: str
    detail: Optional[str] = None


class ProductStatusCheck(EvaluationCheck):
    status: Literal["expired", "damaged", "valid", "unclear"]


class BusinessDecision(BaseModel):
    decision: Literal["accept_claim", "reject_claim", "review"]
    verdict: str
    detail: Optional[str] = None
    reason_codes: list[str] = Field(default_factory=list)


class LinkedImageEvaluationResponse(BaseModel):
    decision: BusinessDecision
    product_status: ProductStatusCheck
    authenticity: EvaluationCheck
    product_match: EvaluationCheck
    query_match: EvaluationCheck
