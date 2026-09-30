"""Recorte da mama quando a mama e PEQUENA no quadro (bug corrigido em 30/09/2026).

Na S2, 20 de 300 imagens do VinDr ficaram sem recorte. Eram todas vistas CC de
mama pequena (6-10% do quadro): o corte "p95" era calculado no quadro inteiro,
caia dentro da mama e apagava o tecido denso, derrubando o contorno para baixo
do limite de 4%.
"""
import numpy as np

from src.preprocess.breast_roi import breast_bbox, breast_mask


def _mama_pequena(h=1000, w=800, seed=0):
    """Meia elipse encostada na parede toracica (esquerda), ~9% do quadro,
    mais densa (clara) perto da parede, e uma etiqueta branca no canto."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, ay, ax = 500, 300, 150
    inside = ((yy - cy) / ay) ** 2 + (xx / ax) ** 2 <= 1
    img = np.zeros((h, w), dtype=np.float32)
    img[inside] = 400 + 1000 * (1 - xx[inside] / ax) + rng.normal(0, 20, inside.sum())
    img[40:60, 650:760] = 4000                    # etiqueta "R-CC" queimada
    return img.astype(np.uint16), inside


def test_mama_pequena_ainda_e_recortada():
    img, inside = _mama_pequena()
    assert inside.mean() < 0.10                   # o caso que falhava
    bb = breast_bbox(img)
    assert bb is not None
    x0, y0, x1, y1 = bb
    assert x0 <= 5 and abs(x1 - 150) <= 10
    assert abs(y0 - 200) <= 15 and abs(y1 - 800) <= 15


def test_recorte_nao_inclui_a_etiqueta():
    img, _ = _mama_pequena()
    x0, y0, x1, y1 = breast_bbox(img)
    assert x1 < 650 and y0 > 60


def test_mascara_cobre_a_mama_inteira_inclusive_a_parte_densa():
    img, inside = _mama_pequena()
    m = breast_mask(img)
    # O corte remove de proposito os 5% de pixels mais claros (marcadores), entao
    # a cobertura fica perto de 95%. Com a regra antiga, este caso dava 47%.
    assert (m & inside).sum() / inside.sum() > 0.90
    assert not (m & ~inside).any()                # nada fora da mama
