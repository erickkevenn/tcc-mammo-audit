"""Geometria: converter uma bounding box em quadrante BI-RADS.

E a peca que permite comparar 'quadrante superior externo' (texto) com uma caixa
(pixel). Nivel 1 do grounding descrito na secao 6.4 do plano -- obrigatorio e barato.

Convencao assumida (garantida por preprocess.breast_roi.canonical_flip):
    a parede toracica esta sempre a ESQUERDA da imagem
    logo: x pequeno = interno (junto ao esterno) ; x grande = externo (axila)
          y pequeno = superior ; y grande = inferior
Em MLO a nocao de 'interno/externo' e mal definida -- retorne apenas vertical.
"""
from __future__ import annotations

QUADRANT = {("upper", "outer"): "UOQ", ("upper", "inner"): "UIQ",
            ("lower", "outer"): "LOQ", ("lower", "inner"): "LIQ"}


def bbox_to_region(bbox, breast_bbox, view: str = "CC",
                   central_frac: float = 0.20) -> dict:
    """Retorna {'quadrant', 'vertical', 'horizontal'} para uma caixa."""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    bx1, by1, bx2, by2 = breast_bbox
    w = max(bx2 - bx1, 1e-6)
    h = max(by2 - by1, 1e-6)
    rx = (cx - bx1) / w
    ry = (cy - by1) / h

    vertical = "upper" if ry < 0.5 - central_frac / 2 else (
        "lower" if ry > 0.5 + central_frac / 2 else None)

    # retroareolar: terco mais distante da parede toracica e faixa central vertical
    if rx > 0.75 and abs(ry - 0.5) < central_frac:
        return {"quadrant": "retroareolar", "vertical": None, "horizontal": None}

    if str(view).upper().startswith("ML") or str(view).upper() == "MLO":
        return {"quadrant": None, "vertical": vertical, "horizontal": None}

    horizontal = "inner" if rx < 0.5 - central_frac / 2 else (
        "outer" if rx > 0.5 + central_frac / 2 else None)
    quad = QUADRANT.get((vertical, horizontal)) if vertical and horizontal else None
    return {"quadrant": quad, "vertical": vertical, "horizontal": horizontal}
