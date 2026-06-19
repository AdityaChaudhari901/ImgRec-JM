from app.models.verify_response import AuthenticityChecks, WebProvenanceCheck


def test_web_provenance_check_defaults():
    c = WebProvenanceCheck(checked=False)
    assert c.full_matches == 0
    assert c.distinct_domains == 0
    assert c.reason is None


def test_authenticity_checks_web_provenance_optional():
    # web_provenance defaults to None so existing callers are unaffected.
    assert "web_provenance" in AuthenticityChecks.model_fields
