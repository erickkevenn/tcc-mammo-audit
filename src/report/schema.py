"""FindingRecord -- a representacao canonica para a qual OS DOIS LADOS convergem.

Sem um esquema unico, o auditor vira um emaranhado de 'if'. Regras de projeto:

 1. TODO campo pode ser None. 'O laudo nao disse' != 'o laudo disse que nao ha'.
    Modele TRES estados: AFFIRMED, NEGATED, ABSENT. A maioria dos erros de
    sistemas deste tipo vem de confundir NEGATED com ABSENT.
 2. TODO achado carrega 'evidence'. Alerta sem rastreabilidade nao e auditoria.
 3. 'modality_section' existe para restringir a comparacao a mamografia
    convencional (DM) ANTES de comparar. Hoje isso filtra sobretudo as
    mencoes a ultrassom/biopsia/axila do INbreast; a distincao DM/CESM e
    suportada de forma generica pelo sectionizer, mas nenhum dataset
    atual do projeto tem laudo com secao CESM.
"""
from __future__ import annotations
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Status(str, Enum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    ABSENT = "absent"


Category = Literal[
    "mass", "suspicious_calcification", "architectural_distortion",
    "focal_asymmetry", "asymmetry", "global_asymmetry",
    "skin_thickening", "skin_retraction", "nipple_retraction",
    "suspicious_lymph_node", "postoperative", "other_suspicious", "no_finding",
]


class Region(BaseModel):
    quadrant: Optional[str] = None      # UOQ|UIQ|LOQ|LIQ|retroareolar|axillary
    vertical: Optional[str] = None      # upper|lower
    horizontal: Optional[str] = None    # outer|inner


class Descriptors(BaseModel):
    mass_shape: Optional[str] = None
    mass_margin: Optional[str] = None
    mass_density: Optional[str] = None
    calc_morphology: Optional[str] = None
    calc_distribution: Optional[str] = None
    size_mm: Optional[float] = None


class Evidence(BaseModel):
    image_id: Optional[str] = None
    bbox: Optional[list[float]] = None            # [x1, y1, x2, y2] no espaco da imagem final
    sentence_span: Optional[list[int]] = None     # [inicio, fim] em caracteres do laudo
    sentence_text: Optional[str] = None


class Finding(BaseModel):
    category: Category
    status: Status = Status.AFFIRMED
    region: Region = Field(default_factory=Region)
    descriptors: Descriptors = Field(default_factory=Descriptors)
    birads: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Evidence = Field(default_factory=Evidence)


class FindingRecord(BaseModel):
    exam_id: str
    laterality: Literal["L", "R"]
    source: Literal["image", "report"]
    findings: list[Finding] = Field(default_factory=list)
    breast_birads: Optional[str] = None
    breast_density: Optional[str] = None
    modality_section: Optional[str] = None        # DM | CESM | US | ...
    language: Optional[Literal["pt", "en"]] = None
    extractor: Optional[str] = None               # rules | llm | cnn


def birads_to_num(value: str | None) -> Optional[float]:
    """'4c' -> 4.3 ; '4a' -> 4.1 ; '4b' -> 4.2 ; '5' -> 5.0. None se ausente.

    Subcategorias viram fracoes para permitir |delta| e ordenacao sem perder 4a/4b/4c.
    """
    if value is None:
        return None
    v = str(value).strip().lower().replace("bi-rads", "").replace("birads", "").strip(" :-")
    sub = {"4a": 4.1, "4b": 4.2, "4c": 4.3}
    if v in sub:
        return sub[v]
    try:
        return float(v[0]) if v and v[0].isdigit() else None
    except (ValueError, IndexError):
        return None


def crosses_action_boundary(a: str | None, b: str | None) -> bool:
    """A fronteira que muda conduta clinica: <= 3 (seguimento) vs >= 4 (biopsia)."""
    na, nb = birads_to_num(a), birads_to_num(b)
    if na is None or nb is None:
        return False
    return (na <= 3) != (nb <= 3)
