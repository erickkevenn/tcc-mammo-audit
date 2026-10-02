"""Metricas do classificador por MAMA (o nivel em que o laudo fala).

Agregacao: media das probabilidades entre as vistas da mama (CC e MLO) e
classe = argmax da media. BI-RADS e ordinal, entao a metrica principal e o
kappa ponderado quadratico; macro-F1 mostra o custo nas classes raras.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .breast_eval import auc
from .stats import quadratic_weighted_kappa

PB = [f"pb{i}" for i in range(5)]
PD = [f"pd{i}" for i in range(3)]


def breast_aggregate(img: pd.DataFrame) -> pd.DataFrame:
    """img: uma linha por imagem com study_id, laterality, birads, density, pb0..4, pd0..2."""
    g = img.groupby(["study_id", "laterality"], sort=True)
    br = g[PB + PD].mean()
    br["birads"] = g.birads.first()
    br["density"] = g.density.first()
    return br.reset_index()


def macro_f1(y, yhat, n_classes: int) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((yhat == c) & (y == c))
        fp = np.sum((yhat == c) & (y != c))
        fn = np.sum((yhat != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def cls_metrics(br: pd.DataFrame) -> dict:
    yb, yd = br.birads.to_numpy(), br.density.to_numpy()
    pb, pd_ = br[PB].to_numpy(), br[PD].to_numpy()
    hb, hd = pb.argmax(1), pd_.argmax(1)
    return {
        "mamas": int(len(br)),
        "birads_qwk": float(quadratic_weighted_kappa(yb, hb, 5)),
        "birads_macro_f1": macro_f1(yb, hb, 5),
        "birads_acc": float((yb == hb).mean()),
        # BI-RADS >= 4 e o limiar de acao clinica (biopsia): e a fronteira que o auditor usa
        "birads_auc_ge4": auc((yb >= 3).astype(int), pb[:, 3] + pb[:, 4]),
        "density_qwk": float(quadratic_weighted_kappa(yd, hd, 3)),
        "density_acc": float((yd == hd).mean()),
    }
