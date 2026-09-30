"""bbox -> quadrante. Em MLO, interno/externo e mal definido: so vertical."""
from src.audit.matcher import bbox_to_region

BREAST = (0, 0, 1000, 2000)     # parede toracica a esquerda (canonicalizado)


def test_upper_outer():
    r = bbox_to_region((800, 200, 900, 300), BREAST, "CC")
    assert r["quadrant"] == "UOQ"


def test_lower_inner():
    r = bbox_to_region((100, 1700, 200, 1800), BREAST, "CC")
    assert r["quadrant"] == "LIQ"


def test_retroareolar():
    r = bbox_to_region((900, 980, 950, 1020), BREAST, "CC")
    assert r["quadrant"] == "retroareolar"


def test_mlo_returns_vertical_only():
    r = bbox_to_region((800, 200, 900, 300), BREAST, "MLO")
    assert r["quadrant"] is None and r["vertical"] == "upper"
