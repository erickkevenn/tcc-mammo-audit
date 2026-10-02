from src.verify.rules import BreastImage, ReportFinding, Thresholds, biopsy_side, verify_breast

T = Thresholds(mass=0.3, calc=0.3, cls_ge4=0.5)
rules = lambda alerts: sorted(a.rule for a in alerts)


def test_biopsy_side_and_unverifiable_categories():
    assert biopsy_side("2") == 0 and biopsy_side("3") == 0
    assert biopsy_side("4a") == 1 and biopsy_side("5") == 1
    assert biopsy_side("0") is None and biopsy_side("6") is None and biopsy_side(None) is None


def test_consistent_cases_give_no_alert():
    assert verify_breast("2", BreastImage(0.05, 0.02, 0.1), [], T) == []
    assert verify_breast("5", BreastImage(0.9, 0.0, 0.8), [], T) == []


def test_v1_suspicious_mass_with_low_category():
    assert rules(verify_breast("2", BreastImage(0.8, 0.0, 0.1), None, T)) == ["V1"]


def test_v1_does_not_fire_for_category_3():
    assert rules(verify_breast("3", BreastImage(0.8, 0.0, 0.2), None, T)) == []


def test_v2_biopsy_category_without_finding():
    assert rules(verify_breast("4b", BreastImage(0.01, 0.01, 0.9), None, T)) == ["V2"]


def test_v4_classifier_on_other_side():
    assert rules(verify_breast("5", BreastImage(0.9, 0.0, 0.1), None, T)) == ["V4"]
    assert rules(verify_breast("1", BreastImage(0.0, 0.0, 0.9), None, T)) == ["V4"]


def test_v3_text_only():
    sus = [ReportFinding("mass", True, "suspicious", "nodulo espiculado")]
    ben = [ReportFinding("suspicious_calcification", True, "benign", "calcificacoes benignas")]
    assert rules(verify_breast("2", None, sus, T)) == ["V3"]
    assert rules(verify_breast("4a", None, ben, T)) == ["V3"]
    assert rules(verify_breast("1", None, ben, T)) == ["V3"]
    assert rules(verify_breast("2", None, ben, T)) == []


def test_categories_0_and_6_are_not_verified():
    img = BreastImage(0.9, 0.9, 0.9)
    assert verify_breast("0", img, [], T) == [] and verify_breast("6", img, [], T) == []
