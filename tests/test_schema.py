"""FindingRecord e conversao de BI-RADS. O ponto mais sensivel: NEGATED != ABSENT."""
from src.report.schema import (Finding, FindingRecord, Status, birads_to_num,
                               crosses_action_boundary)


def test_birads_subcategories_are_ordered():
    assert birads_to_num("4a") < birads_to_num("4b") < birads_to_num("4c") < birads_to_num("5")


def test_birads_tolerates_prefix_and_none():
    assert birads_to_num("BI-RADS 3") == 3.0
    assert birads_to_num(None) is None
    assert birads_to_num("") is None


def test_action_boundary():
    assert crosses_action_boundary("3", "4a") is True
    assert crosses_action_boundary("4a", "5") is False
    assert crosses_action_boundary("1", "2") is False
    assert crosses_action_boundary(None, "5") is False


def test_negated_is_not_absent():
    rec = FindingRecord(exam_id="P1", laterality="L", source="report", findings=[
        Finding(category="suspicious_calcification", status=Status.NEGATED),
    ])
    affirmed = [f for f in rec.findings if f.status == Status.AFFIRMED]
    assert affirmed == []                      # negado nao conta como afirmado
    assert len(rec.findings) == 1              # ...mas tambem nao desaparece


def test_all_descriptor_fields_default_to_none():
    f = Finding(category="mass")
    assert f.descriptors.mass_margin is None and f.birads is None
