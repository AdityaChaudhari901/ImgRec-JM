from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class AIGeneratedCheck(BaseModel):
    is_ai_generated: bool  # convenience boolean: ai_probability >= 0.5
    # Always "probability the image is AI-generated": 0.0 = clearly a real photo,
    # 1.0 = clearly AI-generated/edited. Single, unambiguous direction.
    ai_probability: float
    source: str  # internal | sightengine — which detector produced the verdict
    signals: List[str] = []


class AlignmentCheck(BaseModel):
    score: float  # 0..1 — how well the image matches the user's comment
    aligned: bool
    reason: Optional[str] = None


class ProductMatchCheck(BaseModel):
    matches: bool
    score: float  # 0..1 — how well the image matches the claimed product
    detected_product: Optional[str] = None
    reason: Optional[str] = None


class RecognitionResult(BaseModel):
    """What the model sees and reads in the image (vision + OCR)."""

    scene: Optional[str] = None  # one-line description of what the photo shows
    objects: List[str] = []  # main objects/products recognised in the image
    extracted_text: Optional[str] = None  # all visible text / label OCR


class AuthenticityChecks(BaseModel):
    ai_generated: AIGeneratedCheck
    image_comment_alignment: AlignmentCheck
    product_match: ProductMatchCheck
    other_flags: List[str] = []


class VerifyClaimResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    success: bool
    request_id: str
    order_id: str
    user_id: str
    # Final auditable score computed deterministically from the checks below.
    authenticity_score: float  # 0..1
    # Confidence in the recommended_action (how decisively the score/AI verdict
    # sits in its band). Kaily can require e.g. >= 0.8 before acting unattended.
    decision_confidence: float  # 0..1
    verdict: Literal["authentic", "review", "likely_fraud"]
    recommended_action: Literal["auto_approve", "manual_review", "reject"]
    # What the image actually shows and the text read from it (vision + OCR).
    recognition: RecognitionResult
    checks: AuthenticityChecks
    agent_comment: str
    processed_at: datetime
    model_used: str
