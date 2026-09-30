"""ROI em MONOCHROME1 -- regressao encontrada em 10/09/2026 (S4).

`breast_mask`/`breast_bbox` rodam no array cru, antes da correcao de polaridade.
Em MONOCHROME1 a mama e escura e o fundo claro, entao o Otsu selecionava o fundo
e a bbox voltava o quadro inteiro: 40 de 40 imagens Planmed do VinDr com
crop_area_pct = 1,00, contra 0,14-0,46 nas MONOCHROME2. O sintoma nao e imagem
errada, e perda de resolucao. Este teste tranca a correcao.
"""
import numpy as np

from src.preprocess.breast_roi import breast_bbox
from src.preprocess.roi_input import for_roi


def _fantasma_monochrome2() -> np.ndarray:
    """Fundo escuro com uma 'mama' clara ocupando ~um quarto do quadro."""
    arr = np.zeros((400, 400), dtype=np.uint16)
    arr[100:300, 50:250] = 3000
    return arr


def test_for_roi_nao_toca_em_monochrome2():
    arr = _fantasma_monochrome2()
    saida = for_roi(arr, "MONOCHROME2")
    assert saida is arr


def test_for_roi_inverte_monochrome1_sem_alterar_o_original():
    arr = _fantasma_monochrome2()
    invertido = arr.max() - arr                     # como o DICOM MONOCHROME1 chega
    copia = invertido.copy()
    saida = for_roi(invertido, "MONOCHROME1")

    np.testing.assert_array_equal(invertido, copia)  # nao modifica no lugar
    np.testing.assert_array_equal(saida, arr)        # volta ao original


def test_bbox_igual_nas_duas_polaridades():
    arr = _fantasma_monochrome2()
    invertido = arr.max() - arr

    bb_m2 = breast_bbox(for_roi(arr, "MONOCHROME2"))
    bb_m1 = breast_bbox(for_roi(invertido, "MONOCHROME1"))

    assert bb_m1 is not None and bb_m2 is not None
    assert bb_m1 == bb_m2


def test_bbox_de_monochrome1_nao_cobre_o_quadro_inteiro():
    """O bug: sem for_roi, a caixa vira a imagem toda."""
    arr = _fantasma_monochrome2()
    invertido = arr.max() - arr
    h, w = invertido.shape

    bb = breast_bbox(for_roi(invertido, "MONOCHROME1"))
    area = (bb[2] - bb[0]) * (bb[3] - bb[1]) / float(h * w)
    assert area < 0.5