from app.services.damage_analyzer import VALID_TYPES, normalize_damage


def test_new_tamper_types_accepted():
    for t in ("tamper", "resealed", "missing_component"):
        assert t in VALID_TYPES
        out = normalize_damage({"detected": True, "type": t, "severity": "severe"})
        assert out["type"] == t
