from app.models.verify_response import AIGeneratedCheck
from app.services.authenticity_engine import (
    build_verify_response,
    decision_confidence,
    generate_request_id,
    score_claim,
)
from app.services.dedup_service import DedupResult, DuplicateMatch

NO_AI = AIGeneratedCheck(is_ai_generated=False, ai_probability=0.1, source="internal", signals=[])


def _cross_claim_dup():
    return DedupResult(
        cross_claim_matches=[DuplicateMatch("JM-OTHER", "u_x", "vfy_x", 0, 0.0)]
    )


def _gemini(align=0.9, aligned=True, prod=0.9, matches=True, flags=None):
    return {
        "image_comment_alignment": {"score": align, "aligned": aligned, "reason": "r"},
        "product_match": {"detected_product": "X", "matches": matches, "score": prod, "reason": "r"},
        "other_flags": flags or [],
        "summary": "summary",
    }


def test_high_alignment_and_match_auto_approves():
    score, verdict, action, checks = score_claim(_gemini(0.95, True, 0.95, True), NO_AI)
    assert score >= 0.75
    assert verdict == "authentic"
    assert action == "auto_approve"


def test_low_scores_reject():
    score, verdict, action, _ = score_claim(_gemini(0.1, False, 0.1, False), NO_AI)
    assert score < 0.45
    assert verdict == "likely_fraud"
    assert action == "reject"


def test_mid_scores_go_to_manual_review():
    score, verdict, action, _ = score_claim(_gemini(0.6, True, 0.6, True), NO_AI)
    assert 0.45 <= score < 0.75
    assert verdict == "review"
    assert action == "manual_review"


def test_internal_ai_detection_forces_manual_review_not_reject():
    # Even with perfect alignment/match, a confident *internal* AI verdict must
    # never auto-approve — but must NOT auto-reject either (advisory).
    ai = AIGeneratedCheck(is_ai_generated=True, ai_probability=0.9, source="internal", signals=["x"])
    score, verdict, action = score_claim(_gemini(0.95, True, 0.95, True), ai)[:3]
    assert action == "manual_review"
    assert verdict == "review"


def test_sightengine_ai_detection_rejects():
    ai = AIGeneratedCheck(is_ai_generated=True, ai_probability=0.97, source="sightengine", signals=["x"])
    score, verdict, action = score_claim(_gemini(0.95, True, 0.95, True), ai)[:3]
    assert verdict == "likely_fraud"
    assert action == "reject"


def test_ai_penalty_lowers_score():
    ai = AIGeneratedCheck(is_ai_generated=True, ai_probability=0.5, source="internal", signals=[])
    clean = score_claim(_gemini(0.9, True, 0.9, True), NO_AI)[0]
    penalised = score_claim(_gemini(0.9, True, 0.9, True), ai)[0]
    assert penalised < clean


def test_flags_subtract_from_score():
    no_flags = score_claim(_gemini(0.9, True, 0.9, True, flags=[]), NO_AI)[0]
    with_flags = score_claim(_gemini(0.9, True, 0.9, True, flags=["stock photo", "screenshot"]), NO_AI)[0]
    assert with_flags < no_flags


def test_request_id_prefix():
    assert generate_request_id().startswith("vfy_")


def test_build_verify_response_shape():
    resp = build_verify_response(_gemini(0.9, True, 0.9, True), NO_AI, "JM-1", "u_1")
    assert resp.success is True
    assert resp.order_id == "JM-1"
    assert resp.checks.image_comment_alignment.score == 0.9
    assert resp.recommended_action in {"auto_approve", "manual_review", "reject"}
    assert 0.0 <= resp.decision_confidence <= 1.0
    assert resp.checks.ai_generated.ai_probability == 0.1
    assert isinstance(resp.agent_comment, str) and resp.agent_comment


def test_decision_confidence_higher_for_decisive_scores():
    # A strong auto-approve should be more confident than a borderline one.
    strong = build_verify_response(_gemini(0.99, True, 0.99, True), NO_AI, "JM-1", "u_1")
    borderline = build_verify_response(_gemini(0.76, True, 0.76, True), NO_AI, "JM-1", "u_1")
    assert strong.recommended_action == "auto_approve"
    assert strong.decision_confidence > borderline.decision_confidence


def test_decision_confidence_for_ai_forced_review_uses_probability():
    ai = AIGeneratedCheck(is_ai_generated=True, ai_probability=0.88, source="internal", signals=[])
    resp = build_verify_response(_gemini(0.95, True, 0.95, True), ai, "JM-1", "u_1")
    assert resp.recommended_action == "manual_review"
    assert resp.decision_confidence == 0.88


def test_handles_garbage_gemini_values():
    # Non-numeric scores should not crash; they coerce to 0.
    bad = {"image_comment_alignment": {"score": "nope"}, "product_match": {"score": None}}
    score, verdict, action, _ = score_claim(bad, NO_AI)
    assert score == 0.0
    assert action == "reject"


def test_cross_claim_duplicate_forces_reject_even_with_perfect_scores():
    # A hard, auditable signal (same photo on another claim) overrides everything.
    score, verdict, action, checks = score_claim(
        _gemini(0.99, True, 0.99, True), NO_AI, dedup_result=_cross_claim_dup()
    )
    assert verdict == "likely_fraud"
    assert action == "reject"
    assert score == 0.0
    # The justification is the concrete match (prior order), not an AI score.
    assert any("duplicate" in f.lower() and "JM-OTHER" in f for f in checks.other_flags)


def test_same_claim_duplicate_is_benign_and_does_not_reject():
    same = DedupResult(same_claim_matches=[DuplicateMatch("JM-1", "u_1", "vfy_self", 0, 0.0)])
    score, verdict, action, _ = score_claim(
        _gemini(0.95, True, 0.95, True), NO_AI, dedup_result=same
    )
    assert action == "auto_approve"


def test_hard_duplicate_decision_confidence_is_max():
    resp = build_verify_response(
        _gemini(0.99, True, 0.99, True), NO_AI, "JM-1", "u_1", dedup_result=_cross_claim_dup()
    )
    assert resp.recommended_action == "reject"
    assert resp.decision_confidence == 1.0
    assert "JM-OTHER" in resp.agent_comment


from app.services.web_provenance import WebProvenanceResult


def _web(full=0, partial=0, domains=0, checked=True):
    return WebProvenanceResult(
        full_match_count=full, partial_match_count=partial,
        distinct_domains=domains, checked=checked,
    )


def test_web_match_two_domains_forces_reject():
    score, verdict, action, checks = score_claim(
        _gemini(0.99, True, 0.99, True), NO_AI, web_result=_web(full=2, domains=2)
    )
    assert verdict == "likely_fraud"
    assert action == "reject"
    assert score == 0.0
    assert checks.web_provenance is not None and checks.web_provenance.distinct_domains == 2


def test_web_match_single_domain_is_soft_penalty_only():
    clean = score_claim(_gemini(0.9, True, 0.9, True), NO_AI, web_result=_web())[0]
    soft = score_claim(_gemini(0.9, True, 0.9, True), NO_AI, web_result=_web(full=1, domains=1))[0]
    assert soft < clean
    assert soft > 0.0  # not a hard reject


def test_web_unchecked_leaves_score_unchanged():
    base = score_claim(_gemini(0.9, True, 0.9, True), NO_AI)[0]
    unchecked = score_claim(
        _gemini(0.9, True, 0.9, True), NO_AI, web_result=_web(checked=False)
    )[0]
    assert unchecked == base


def test_score_out_of_100_matches_authenticity_score():
    resp = build_verify_response(_gemini(0.9, True, 0.9, True), NO_AI, "JM-1", "u_1")
    assert resp.score_out_of_100 == round(resp.authenticity_score * 100)


def test_web_hard_decision_confidence_is_max():
    resp = build_verify_response(
        _gemini(0.99, True, 0.99, True), NO_AI, "JM-1", "u_1", web_result=_web(full=3, domains=3)
    )
    assert resp.recommended_action == "reject"
    assert resp.decision_confidence == 1.0
