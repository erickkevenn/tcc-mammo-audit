"""Plano de splits do VinDr para o detector (to_yolo).

Protege as duas correcoes de 30/09/2026: validacao separada do teste oficial,
por estudo, e imagens normais presentes em validacao e teste.
"""
import math

import numpy as np
import pandas as pd

from src.data.vindr_splits import image_table, plan_images, summary

CLASSES = ["mass", "architectural_distortion"]


def _findings(n_studies=200, seed=0):
    """Achados sinteticos no formato de load_findings(): 4 imagens por estudo."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_studies):
        split = "test" if s % 5 == 0 else "training"
        for v in range(4):
            img = f"S{s:04d}_I{v}"
            kind = rng.choice(["neg", "pos", "calc"], p=[0.8, 0.15, 0.05])
            if kind == "neg":
                rows.append(dict(study_id=f"S{s:04d}", image_id=img, split=split,
                                 has_box=False, categories_mapped=[],
                                 xmin=math.nan, ymin=math.nan, xmax=math.nan, ymax=math.nan))
            else:
                cat = ["mass"] if kind == "pos" else ["suspicious_calcification"]
                rows.append(dict(study_id=f"S{s:04d}", image_id=img, split=split,
                                 has_box=True, categories_mapped=cat,
                                 xmin=10.0, ymin=20.0, xmax=110.0, ymax=120.0))
    return pd.DataFrame(rows)


def _plan(**kw):
    return plan_images(image_table(_findings(), CLASSES), **kw)


def test_nenhum_estudo_em_dois_splits():
    plan = _plan()
    assert (plan.groupby("study_id").split.nunique() == 1).all()


def test_teste_oficial_fica_intocado_e_validacao_sai_do_treino():
    df = _findings()
    oficial_teste = set(df[df.split == "test"].study_id)
    plan = _plan()
    assert set(plan[plan.split == "test"].study_id) <= oficial_teste
    assert not (set(plan[plan.split == "val"].study_id) & oficial_teste)
    assert not (set(plan[plan.split == "training"].study_id) & oficial_teste)
    assert (plan.split == "val").any()


def test_validacao_e_teste_recebem_todas_as_normais_por_padrao():
    img = image_table(_findings(), CLASSES)
    plan = plan_images(img)
    test_studies = set(img[img.official_split == "test"].study_id)
    negs_teste = img[(img.role == "neg") & img.study_id.isin(test_studies)]
    assert (plan[plan.split == "test"].role == "neg").sum() == len(negs_teste)
    assert (plan[plan.split == "val"].role == "neg").any()


def test_negativas_de_treino_seguem_a_razao_pedida():
    plan = _plan(train_neg_ratio=0.25)
    tr = plan[plan.split == "training"]
    n_pos, n_neg = (tr.role == "pos").sum(), (tr.role == "neg").sum()
    assert n_neg == round(0.25 * n_pos)


def test_imagem_so_com_classe_fora_do_detector_fica_de_fora():
    img = image_table(_findings(), CLASSES)
    assert (img.role == "out").any()
    plan = plan_images(img)
    assert not (plan.role == "out").any()
    assert all(len(b) == 0 for b in plan[plan.role == "neg"].boxes)
    assert all(len(b) > 0 for b in plan[plan.role == "pos"].boxes)


def test_plano_e_deterministico():
    a, b = _plan(), _plan()
    pd.testing.assert_frame_equal(a[["image_id", "split", "role"]], b[["image_id", "split", "role"]])


def test_summary_tem_as_tres_linhas():
    t = summary(_plan())
    assert list(t.index) == ["training", "val", "test"]
    assert {"pos", "neg", "caixas", "estudos"} <= set(t.columns)
