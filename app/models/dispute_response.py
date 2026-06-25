from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Decision = Literal["approve", "reject", "agent"]
Route = Literal["auto", "agent"]
RefundType = Literal["price_difference", "full_selling_price", "none"]
CategorySource = Literal["provided", "description", "notes", "disposition", "none"]


class RefundResult(BaseModel):
    eligible: bool = False
    amount: float = 0.0
    type: RefundType = "none"
    assign_to_mpt: bool = False
    seller_debit: bool = False


class DisputeResponse(BaseModel):
    success: bool = True
    request_id: str
    order_tracking_id: str
    category: Optional[str] = None
    category_source: CategorySource = "none"
    decision: Decision
    route: Route
    agent_flags: List[str] = Field(default_factory=list)
    refund: RefundResult = Field(default_factory=RefundResult)
    recommendation: str = ""
    confidence: float = 0.0
    observations: Dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_used: str = ""
