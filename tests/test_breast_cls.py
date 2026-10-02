"""Classificador por mama: tabela, splits, pesos e metricas (sem GPU)."""
import numpy as np
import pandas as pd
import pytest

from src.data.vindr_breast import check_breast_labels, cls_table, sample_weights
from src.data.vindr_splits import VAL_SALT, hash01
from src.eval.cls_metrics import PB, PD, breast_aggregate, cls_metrics, macro_f1


def _breast_csv(n_studies=80):
    rows = []
    for s in range(n_studies):
        split = "test" if s % 5 == 0 else "training"
        for lat, b in (("L", 1 + s % 5), ("R", 1 + (s + 2) % 5)):
            for view in ("CC", "MLO"):
                rows.append(dict(study_id=f"S{s:03d}", image_id=f"S{s:03d}{lat}{view}",
                                 laterality=lat, view_position=view,
                                 breast_birads=f"BI-RADS {b}",
                                 breast_density=f"DENSITY {'ABCD'[s % 4]}", split=split))
    return pd.DataFrame(rows)


def test_rotulos_convertidos_e_densidade_a_fundida_com_b():
    t = cls_table(_breast_csv())
    assert set(t.birads) == {0, 1, 2, 3, 4}
    assert set(t.density) == {0, 1, 2}
    assert (t[t.study_id == "S000"].density == 0).all()          # A -> 0
    assert (t[t.study_id == "S001"].density == 0).all()          # B -> 0


def test_validacao_usa_o_mesmo_hash_do_detector():
    t = cls_table(_breast_csv())
    for sid, g in t.groupby("study_id"):
        oficial_teste = int(sid[1:]) % 5 == 0
        esperado = "test" if oficial_teste else ("val" if hash01(sid, VAL_SALT) < 0.10 else "training")
        assert set(g.split) == {esperado}


def test_rotulo_consistente_entre_vistas():
    t = cls_table(_breast_csv())
    assert check_breast_labels(t) == 0
    t.loc[0, "birads"] = 4
    assert check_breast_labels(t) == 1


def test_peso_maior_para_classe_rara():
    t = cls_table(_breast_csv()).iloc[:-8]                       # desequilibra
    w = sample_weights(t)
    freq = t.birads.value_counts()
    rara, comum = freq.idxmin(), freq.idxmax()
    assert w[t.birads == rara].iloc[0] > w[t.birads == comum].iloc[0]
    assert (sample_weights(t, power=0) == 1).all()


def test_densidade_invalida_falha_alto():
    df = _breast_csv()
    df.loc[0, "breast_density"] = "DENSITY X"
    with pytest.raises(ValueError):
        cls_table(df)


def _img_preds():
    rows = []
    for s, (b, d) in enumerate([(0, 0), (1, 1), (3, 2), (4, 1)]):
        for view, conf in (("CC", 0.9), ("MLO", 0.5)):
            pb = np.full(5, (1 - conf) / 4); pb[b] = conf
            pd_ = np.full(3, (1 - conf) / 2); pd_[d] = conf
            rows.append({"study_id": f"S{s}", "laterality": "L", "birads": b, "density": d,
                         **dict(zip(PB, pb)), **dict(zip(PD, pd_))})
    return pd.DataFrame(rows)


def test_mama_e_a_media_das_vistas_e_metricas_perfeitas():
    br = breast_aggregate(_img_preds())
    assert len(br) == 4
    assert np.isclose(br.loc[0, "pb0"], (0.9 + 0.5) / 2)
    m = cls_metrics(br)
    assert m["birads_qwk"] == 1.0 and m["birads_acc"] == 1.0 and m["density_acc"] == 1.0
    assert m["birads_auc_ge4"] == 1.0


def test_macro_f1():
    assert macro_f1([0, 0, 1, 1], [0, 0, 1, 1], 2) == 1.0
    assert np.isclose(macro_f1([0, 0, 1, 1], [0, 1, 1, 1], 2), (2 / 3 + 0.8) / 2)
