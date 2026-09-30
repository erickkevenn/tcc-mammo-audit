"""Avaliacao do detector nos dois niveis que importam para o TCC.

1. LESAO (FROC): sensibilidade por falso positivo por imagem. Usa froc_curve.
2. MAMA: o nivel em que o auditor trabalha, porque o laudo descreve por mama.
   Cada mama recebe a MAIOR confianca entre suas vistas (CC e MLO); a mama e
   positiva se alguma vista tem caixa da classe avaliada. Mede AUC e
   sensibilidade quando so uma fracao fixa das mamas normais gera alerta.

IC 95% por bootstrap PERCENTIL reamostrando EXAMES (nunca imagens soltas),
todas as metricas na mesma reamostragem. Percentil em vez de BCa por custo:
a FROC e recalculada a cada reamostragem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .froc import OPERATING_POINTS, froc_curve


def yolo_to_xyxy(line: str, w: int, h: int) -> tuple[int, tuple[float, float, float, float]]:
    """'cls cx cy bw bh' normalizado -> (cls, (x1, y1, x2, y2)) em pixels."""
    cls, cx, cy, bw, bh = line.split()
    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return int(cls), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def build_units(images: pd.DataFrame, preds: dict, gts: dict) -> list[dict]:
    """Um item por EXAME, com todas as suas imagens.

    images: colunas image_id, study_id, laterality
    preds : {image_id: [(bbox, score), ...]}   gts: {image_id: [bbox, ...]}
    """
    units = []
    for study_id, g in images.groupby("study_id", sort=True):
        units.append({"study_id": study_id, "images": [
            {"image_id": r.image_id, "laterality": r.laterality,
             "preds": preds.get(r.image_id, []), "gts": gts.get(r.image_id, [])}
            for r in g.itertuples(index=False)]})
    return units


def lesion_froc(units: list[dict], criterion: str = "iou", iou_thr: float = 0.5) -> dict:
    """FROC sobre as unidades. A chave leva o indice da unidade para que um exame
    sorteado duas vezes no bootstrap conte como dois exames, sem colidir."""
    preds, gts = {}, {}
    for ui, u in enumerate(units):
        for im in u["images"]:
            k = f"{ui}:{im['image_id']}"
            preds[k] = im["preds"]
            gts[k] = im["gts"]
    return froc_curve(preds, gts, criterion=criterion, iou_thr=iou_thr)


def breast_table(units: list[dict]) -> pd.DataFrame:
    rows = []
    for ui, u in enumerate(units):
        by_side: dict[str, dict] = {}
        for im in u["images"]:
            d = by_side.setdefault(im["laterality"], {"label": 0, "score": 0.0})
            d["label"] = max(d["label"], int(len(im["gts"]) > 0))
            if im["preds"]:
                d["score"] = max(d["score"], max(s for _, s in im["preds"]))
        for side, d in by_side.items():
            rows.append({"unit": ui, "laterality": side, **d})
    return pd.DataFrame(rows, columns=["unit", "laterality", "label", "score"])


def auc(labels, scores) -> float:
    """AUC ROC por Mann-Whitney (empates contam meio)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sens_at_spec(labels, scores, spec: float = 0.90) -> float:
    """Sensibilidade no limiar em que uma fracao `spec` das mamas normais fica
    sem alerta (ou seja, 1-spec delas gera alerta falso)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(neg, spec)
    return float((pos > thr).mean())


def all_metrics(units: list[dict], criterion: str = "iou", iou_thr: float = 0.5) -> dict:
    fr = lesion_froc(units, criterion, iou_thr)
    out = {f"lesao_{k}": v for k, v in fr["operating_points"].items()}
    out["lesao_pAUC_0.125-4"] = fr["partial_auc_0.125_4"]
    # a pAUC so vale se a curva chega a 4 FP/imagem; se parar antes, a area fica truncada
    out["lesao_fppi_max"] = float(fr["fppi"][-1]) if fr["fppi"] else 0.0
    bt = breast_table(units)
    out["mama_AUC"] = auc(bt.label, bt.score)
    out["mama_sens@spec90"] = sens_at_spec(bt.label, bt.score, 0.90)
    out["mama_sens@spec80"] = sens_at_spec(bt.label, bt.score, 0.80)
    return out


def evaluate(units: list[dict], n_boot: int = 1000, criterion: str = "iou",
             iou_thr: float = 0.5, seed: int = 20260819) -> pd.DataFrame:
    """Tabela metrica x (ponto, IC 2,5%, IC 97,5%)."""
    point = all_metrics(units, criterion, iou_thr)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(units), size=len(units))
        boots.append(all_metrics([units[i] for i in idx], criterion, iou_thr))
    b = pd.DataFrame(boots)
    return pd.DataFrame({"valor": pd.Series(point),
                         "ic_lo": b.quantile(0.025), "ic_hi": b.quantile(0.975)})


def counts(units: list[dict]) -> dict:
    bt = breast_table(units)
    n_img = sum(len(u["images"]) for u in units)
    n_les = sum(len(im["gts"]) for u in units for im in u["images"])
    return {"exames": len(units), "imagens": n_img, "lesoes": n_les,
            "mamas_pos": int(bt.label.sum()), "mamas_neg": int((bt.label == 0).sum())}


__all__ = ["OPERATING_POINTS", "yolo_to_xyxy", "build_units", "lesion_froc", "breast_table",
           "auc", "sens_at_spec", "all_metrics", "evaluate", "counts"]
