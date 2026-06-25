from app.db.models import ClaimDecision


def test_dispute_columns_exist():
    cols = {c.name for c in ClaimDecision.__table__.columns}
    assert {"category", "category_source", "decision", "route", "agent_flags", "refund"} <= cols


def test_endpoint_constraint_allows_dispute():
    # The /dispute endpoint persists endpoint="dispute"; the CHECK constraint must
    # permit it or every dispute audit write fails in Postgres.
    constraints = [c for c in ClaimDecision.__table__.constraints
                   if getattr(c, "name", "") == "ck_claim_decisions_endpoint"]
    assert constraints, "endpoint check constraint missing"
    assert "dispute" in str(constraints[0].sqltext)
