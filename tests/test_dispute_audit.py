from app.models.dispute_response import DisputeResponse, RefundResult
from app.services.audit_service import _derive_routing, _downgrade


def _resp(decision="approve", route="auto"):
    return DisputeResponse(
        request_id="dsp_1", order_tracking_id="JM-1", category="mrp_abuse",
        category_source="provided", decision=decision, route=route,
        refund=RefundResult(eligible=True, amount=4.0, type="price_difference"),
        recommendation="ok", confidence=0.9, observations={}, model_used="m",
    )


def test_derive_routing_dispute_auto():
    action, status, prio, routed = _derive_routing("dispute", _resp())
    assert action == "approve"
    assert routed == "auto"


def test_derive_routing_dispute_agent():
    _, _, _, routed = _derive_routing("dispute", _resp(decision="agent", route="agent"))
    assert routed == "human"


def test_downgrade_forces_agent():
    out = _downgrade("dispute", _resp())
    assert out.decision == "agent"
    assert out.route == "agent"
