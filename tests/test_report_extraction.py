"""Extrator de regras e seccionador.

INBREAST_PT e verbatim (achados) do formato REAL do seu corpus (INbreast, PT).
SYNTHETIC_EN e um exemplo CONSTRUIDO (nao corresponde a nenhum dataset do
projeto -- o CDD-CESM, que tinha esse formato de laudo em ingles com secao
DM/OPINION/CESM, foi removido do escopo em 24/08/2026). Mantido para testar,
de forma generica, o seccionador em ingles e a separacao de secao CESM.
"""
from src.report.extract_rules import RuleExtractor
from src.report.schema import Status
from src.report.sectionizer import auditable_sections, split_by_modality

SYNTHETIC_EN = (
    "PATIENT NO.1\n"
    "DIGITALIZED LOW DOSE SOFT TISSUE MAMMOGRAPHY REVEALED:\n"
    "ACR C: Heterogeneously dense breasts.\n"
    "Left Breast:\n"
    "Upper architectural distortion is seen.\n"
    "No suspicious microcalcifications.\n"
    "OPINION:\n"
    "Left Breast:\n"
    "Upper architectural distortion (BIRADS 4).\n"
    "CONTRAST ENHANCED SPECTRAL MAMMOGRAPHY REVEALED:\n"
    "Left Breast:\n"
    "Upper heterogeneous non mass enhancement (BIRADS 4).\n"
)

INBREAST_PT = (
    "O estudo imagiologico documenta nodulo com distorcao do estroma, localizado no QII "
    "da mama direita com 2 cm de diametro, sem microcalcificacoes associadas.\n"
    "Axila negativa.\n"
    "Alteracoes com suspeicao elevada de malignidade - Bi-Rads - 4c\n"
)


def test_modality_split_finds_dm_and_cesm():
    parts = split_by_modality(SYNTHETIC_EN)
    assert "DM" in parts and "CESM" in parts
    assert parts["DM"][1] <= parts["CESM"][0]


def test_cesm_section_is_discarded():
    text = " ".join(s.text for s in auditable_sections(SYNTHETIC_EN, "en", "DM"))
    assert "non mass enhancement" not in text


def test_ultrasound_and_axilla_lines_are_discarded_pt():
    text = " ".join(s.text for s in auditable_sections(INBREAST_PT, "pt", "DM"))
    assert "Axila" not in text


def test_extracts_distortion_and_negated_calcification():
    rec = RuleExtractor("en").extract(SYNTHETIC_EN, "P1", "L")
    cats = {(f.category, f.status) for f in rec.findings}
    assert ("architectural_distortion", Status.AFFIRMED) in cats
    assert ("suspicious_calcification", Status.NEGATED) in cats


def test_extracts_density_and_breast_birads():
    rec = RuleExtractor("en").extract(SYNTHETIC_EN, "P1", "L")
    assert rec.breast_density == "C"
    assert rec.breast_birads == "4"


def test_portuguese_extraction_with_size_and_birads():
    rec = RuleExtractor("pt").extract(INBREAST_PT, "case1", "R")
    assert rec.breast_birads == "4c"
    cats = {f.category for f in rec.findings}
    assert "mass" in cats or "architectural_distortion" in cats
    sizes = [f.descriptors.size_mm for f in rec.findings if f.descriptors.size_mm]
    assert 20.0 in sizes                        # '2 cm' -> 20 mm


def test_every_finding_carries_evidence():
    rec = RuleExtractor("en").extract(SYNTHETIC_EN, "P1", "L")
    assert all(f.evidence.sentence_text for f in rec.findings)
