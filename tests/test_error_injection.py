"""Injecao de erro: SEMPRE condicionada ao contexto (regra do ReXErr)."""
from src.eval.error_injection import PERTURBATIONS, applicable, build_benchmark, inject

REPORT = (
    "DIGITALIZED LOW DOSE SOFT TISSUE MAMMOGRAPHY REVEALED:\n"
    "ACR C: Heterogeneously dense breasts.\n"
    "Left Breast:\n"
    "An irregular spiculated mass is seen in the upper outer quadrant.\n"
    "Right Breast:\n"
    "No suspicious microcalcifications.\n"
    "OPINION:\nLeft Breast: Suspicious mass (BIRADS 5).\n"
)
NORMAL = "Right Breast:\nNormal breast examination (BIRADS 1).\n"


def test_omission_removes_the_finding_sentence():
    out = inject(REPORT, "omission")
    assert out is not None and "spiculated mass" not in out.text


def test_side_swap_inverts_sides():
    out = inject(REPORT, "side_swap")
    assert out is not None
    assert out.text.index("Right Breast") < out.text.index("Left Breast")


def test_severity_change_lowers_birads():
    out = inject(REPORT, "severity_change")
    assert out is not None and "BIRADS 5" not in out.text


def test_descriptor_swap_changes_margin():
    out = inject(REPORT, "descriptor_swap")
    assert out is not None and "circumscribed" in out.text.lower()


def test_density_change():
    out = inject(REPORT, "density_change")
    assert out is not None and "ACR C" not in out.text


def test_context_conditioning_blocks_impossible_errors():
    assert applicable(NORMAL, "descriptor_swap") is False
    assert inject(NORMAL, "descriptor_swap") is None
    assert applicable(NORMAL, "omission") is False


def test_every_perturbation_maps_to_an_alert_code():
    for kind, spec in PERTURBATIONS.items():
        assert spec["alert"].startswith("A")


def test_benchmark_includes_clean_cases():
    cases = build_benchmark({"P1": REPORT, "P2": NORMAL}, per_report=2)
    kinds = [c["kind"] for c in cases]
    assert "clean" in kinds
    assert sum(1 for k in kinds if k == "clean") == 2
