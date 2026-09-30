"""Avaliacao por lesao (FROC) e por mama, com bootstrap por exame."""
import numpy as np
import pandas as pd

from src.eval.breast_eval import (auc, breast_table, build_units, counts, evaluate,
                                  lesion_froc, sens_at_spec, yolo_to_xyxy)


def test_yolo_to_xyxy_volta_para_pixels():
    cls, box = yolo_to_xyxy("0 0.5 0.25 0.1 0.2", w=640, h=1024)
    assert cls == 0
    assert np.allclose(box, (288.0, 153.6, 352.0, 358.4))


def _units():
    imgs = pd.DataFrame([
        {"image_id": "a_cc", "study_id": "A", "laterality": "L"},
        {"image_id": "a_mlo", "study_id": "A", "laterality": "L"},
        {"image_id": "a_r", "study_id": "A", "laterality": "R"},
        {"image_id": "b_cc", "study_id": "B", "laterality": "L"},
    ])
    gt = (100, 100, 200, 200)
    gts = {"a_cc": [gt], "a_mlo": [gt]}
    preds = {"a_cc": [((100, 100, 200, 200), 0.9)],          # acerto
             "a_mlo": [((500, 500, 600, 600), 0.4)],          # errou o lugar
             "a_r": [((10, 10, 20, 20), 0.2)],                # FP em mama normal
             "b_cc": []}
    return build_units(imgs, preds, gts)


def test_mama_recebe_a_maior_confianca_entre_as_vistas():
    bt = breast_table(_units()).set_index(["unit", "laterality"])
    assert bt.loc[(0, "L")].label == 1 and bt.loc[(0, "L")].score == 0.9
    assert bt.loc[(0, "R")].label == 0 and bt.loc[(0, "R")].score == 0.2
    assert bt.loc[(1, "L")].label == 0 and bt.loc[(1, "L")].score == 0.0


def test_froc_conta_lesoes_e_imagens_inclusive_normais():
    fr = lesion_froc(_units())
    assert fr["n_lesions"] == 2 and fr["n_images"] == 4
    assert max(fr["sensitivity"]) == 0.5                      # 1 de 2 lesoes


def test_auc_e_sensibilidade_em_especificidade():
    assert auc([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2]) == 1.0
    assert auc([1, 0], [0.5, 0.5]) == 0.5
    assert sens_at_spec([1, 1, 0, 0, 0, 0], [0.9, 0.1, 0.1, 0.2, 0.3, 0.4], 0.75) == 0.5


def test_bootstrap_nao_colide_exame_sorteado_duas_vezes():
    u = _units()
    fr = lesion_froc([u[0], u[0]])
    assert fr["n_images"] == 6 and fr["n_lesions"] == 4


def test_evaluate_devolve_ic_que_contem_o_ponto():
    tab = evaluate(_units() * 5, n_boot=50)
    assert {"valor", "ic_lo", "ic_hi"} <= set(tab.columns)
    ok = tab.dropna()
    assert ((ok.ic_lo <= ok.valor + 1e-9) & (ok.valor <= ok.ic_hi + 1e-9)).all()
    assert counts(_units()) == {"exames": 2, "imagens": 4, "lesoes": 2, "mamas_pos": 1, "mamas_neg": 2}
