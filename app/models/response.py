from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from app.models.verify_response import AIGeneratedCheck


class OCRResult(BaseModel):
    manufacture_date: Optional[str] = None
    expiry_date: Optional[str] = None
    batch_no: Optional[str] = None
    days_since_expiry: Optional[int] = None
    raw_text: Optional[str] = None


class DamageResult(BaseModel):
    detected: bool = False
    type: Optional[
        Literal[
            "crushed_packaging",
            "tear",
            "broken_seal",
            "leakage",
            "dent",
            "discoloration",
            "mold",
        ]
    ] = None
    severity: Optional[Literal["minor", "moderate", "severe"]] = None
    description: Optional[str] = None


class ActionResult(BaseModel):
    type: Literal["initiate_refund", "initiate_exchange", "no_action"]
    message: str
    refund_eligible: bool
    priority: Literal["high", "medium", "low"]


class ScanResponse(BaseModel):
    # `model_used` collides with Pydantic's protected "model_" namespace; opt out.
    model_config = ConfigDict(protected_namespaces=())

    success: bool
    request_id: str
    order_id: str
    user_id: str
    status: Literal["expired", "damaged", "valid", "unclear"]
    confidence: float
    ocr: OCRResult
    damage: DamageResult
    ai_generated: Optional[AIGeneratedCheck] = None
    action: ActionResult
    processed_at: datetime
    model_used: str
