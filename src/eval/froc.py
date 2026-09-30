"""FROC -- a metrica primaria. Sensibilidade em nivel de LESAO por FP/imagem.

Por que FROC e nao mAP: opera em nivel de lesao, distingue 'detectou 1 de 3' de
'detectou 3 de 3', e o eixo x (FP por imagem) e diretamente interpretavel como
carga de trabalho clinica. mAP e insensivel ao NUMERO ABSOLUTO de FP por imagem.

Recomendacao do Metrics Reloaded (Nature Methods 2024): combine uma metrica de
contagem (F-beta) com uma multi-limiar (AP para padronizacao OU FROC para
interpretabilidade), e declare o criterio de localizacao.

Criterio de acerto:
  massas / distorcoes / assimetrias -> IoU >= 0,5
  calcificacoes                     -> CENTRO dentro da caixa de referencia
                                       (IoU e instavel em objeto minusculo)
"""
from __future__ import annotations
import numpy as np

OPERATING_POINTS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def center_in_box(pred, gt) -> bool:
    cx, cy = (pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0
    return gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]


def froc_curve(preds: dict, gts: dict, criterion: str = "iou",
               iou_thr: float = 0.5) -> dict:
    """preds: {image_id: [(bbox, score), ...]}   gts: {image_id: [bbox, ...]}

    Retorna curva (fppi, sensibilidade) e os pontos de operacao.
    """
    n_images = max(len(gts), 1)
    n_lesions = sum(len(v) for v in gts.values())

    rows = []
    for img, dets in preds.items():
        for bbox, score in dets:
            rows.append((score, img, bbox))
    rows.sort(key=lambda r: -r[0])

    matched: dict[str, set[int]] = {k: set() for k in gts}
    tp_hits, fp_count = [], []
    tp = fp = 0
    for score, img, bbox in rows:
        hit = -1
        for gi, gt in enumerate(gts.get(img, [])):
            if gi in matched.get(img, set()):
                continue
            ok = (center_in_box(bbox, gt) if criterion == "center"
                  else iou(bbox, gt) >= iou_thr)
            if ok:
                hit = gi
                break
        if hit >= 0:
            matched.setdefault(img, set()).add(hit)
            tp += 1
        else:
            fp += 1
        tp_hits.append(tp)
        fp_count.append(fp)

    sens = np.array(tp_hits, dtype=float) / max(n_lesions, 1)
    fppi = np.array(fp_count, dtype=float) / n_images

    ops = {}
    for point in OPERATING_POINTS:
        idx = np.searchsorted(fppi, point, side="right") - 1
        ops[f"sens@{point}fppi"] = float(sens[idx]) if idx >= 0 else 0.0

    return {
        "fppi": fppi.tolist(), "sensitivity": sens.tolist(),
        "n_images": n_images, "n_lesions": n_lesions,
        "operating_points": ops,
        "partial_auc_0.125_4": float(_partial_auc(fppi, sens, 0.125, 4.0)),
    }


def _partial_auc(fppi, sens, lo, hi) -> float:
    """FROC parcial normalizada -> valor em [0,1], comparavel entre datasets.
    A area sob a FROC completa NAO e limitada a [0,1] e e dominada por casos
    com muitos FP; use sempre a versao parcial em faixa clinica."""
    if len(fppi) == 0:
        return 0.0
    m = (fppi >= lo) & (fppi <= hi)
    if not m.any():
        return 0.0
    return float(np.trapezoid(sens[m], fppi[m]) / (hi - lo))
