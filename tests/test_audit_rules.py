"""Regras A1-A8. Cada teste corresponde a uma linha da taxonomia do plano."""
from src.audit.rules import Auditor
from src.report.schema import (Descriptors, Evidence, Finding, FindingRecord,
                               Region, Status)


def img(findings=None, birads=None, density=None):
    return FindingRecord(exam_id="P1", laterality="L", source="image",
                         findings=findings or [], breast_birads=birads,
                         breast_density=density)


def rep(findings=None, birads=None, density=None):
    return FindingRecord(exam_id="P1", laterality="L", source="report",
                         findings=findings or [], breast_birads=birads,
                         breast_density=density)


A = Auditor()


def test_a1_omission():
    alerts = A.audit(
        img([Finding(category="mass", confidence=0.9, birads="5")]),
        rep([]),
    )
    codes = [a.code for a in alerts]
    assert "A1" in codes
    assert max(a.severity for a in alerts if a.code == "A1") == 5


def test_low_confidence_detection_does_not_alert():
    alerts = A.audit(img([Finding(category="mass", confidence=0.05)]), rep([]))
    assert "A1" not in [a.code for a in alerts]


def test_a1_flags_explicit_negation():
    alerts = A.audit(
        img([Finding(category="suspicious_calcification", confidence=0.9)]),
        rep([Finding(category="suspicious_calcification", status=Status.NEGATED)]),
    )
    a1 = [a for a in alerts if a.code == "A1"]
    assert a1 and a1[0].evidence["explicitly_negated"] is True


def test_a2_unsupported_claim():
    alerts = A.audit(img([]), rep([Finding(category="mass")]))
    assert "A2" in [a.code for a in alerts]


def test_no_alert_when_consistent():
    f_img = Finding(category="mass", confidence=0.9,
                    region=Region(quadrant="UOQ"),
                    descriptors=Descriptors(mass_margin="spiculated"))
    f_rep = Finding(category="mass", region=Region(quadrant="UOQ"),
                    descriptors=Descriptors(mass_margin="spiculated"))
    assert A.audit(img([f_img]), rep([f_rep])) == []


def test_a5_descriptor_mismatch():
    alerts = A.audit(
        img([Finding(category="mass", confidence=0.9,
                     descriptors=Descriptors(mass_margin="spiculated"))]),
        rep([Finding(category="mass",
                     descriptors=Descriptors(mass_margin="circumscribed"),
                     evidence=Evidence(sentence_text="circumscribed mass"))]),
    )
    assert "A5" in [a.code for a in alerts]


def test_a4_location_mismatch():
    alerts = A.audit(
        img([Finding(category="mass", confidence=0.9, region=Region(quadrant="UOQ"))]),
        rep([Finding(category="mass", region=Region(quadrant="LIQ"))]),
    )
    assert "A4" in [a.code for a in alerts]


def test_unspecified_region_matches_anything():
    alerts = A.audit(
        img([Finding(category="mass", confidence=0.9, region=Region(quadrant="UOQ"))]),
        rep([Finding(category="mass", region=Region())]),
    )
    assert "A4" not in [a.code for a in alerts]


def test_a6_crossing_action_boundary_is_urgent():
    alerts = [a for a in A.audit(img(birads="5"), rep(birads="2")) if a.code == "A6"]
    assert alerts and alerts[0].severity == 4


def test_a6_small_difference_is_silent():
    assert [a for a in A.audit(img(birads="4a"), rep(birads="4b")) if a.code == "A6"] == []


def test_a7_density():
    assert "A7" in [a.code for a in A.audit(img(density="D"), rep(density="B"))]


def test_a8_internal_inconsistency():
    r = rep([Finding(category="mass",
                     descriptors=Descriptors(mass_margin="spiculated"),
                     evidence=Evidence(sentence_text="spiculated mass"))], birads="2")
    assert "A8" in [a.code for a in A.audit_report_internal(r)]
