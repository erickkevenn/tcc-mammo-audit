"""Caminho de calcificacoes: tiles em resolucao quase nativa.

Justificativa dura: ~19% das caixas anotadas tem menos de 48x48 px no original;
a 640x640 viram ~15 px -- e por isso que calcificacao da mAP50 < 0,12 nos
detectores publicados. Onde se treinou com janelas de 256x256 em alta resolucao,
calcificacao (F1 0,843) saiu MELHOR que massa (F1 0,739).
"""
from __future__ import annotations
import numpy as np


def tile_positions(h: int, w: int, size: int = 512, stride: int = 448):
    ys = list(range(0, max(h - size, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - size, 0) + 1, stride)) or [0]
    if ys[-1] + size < h:
        ys.append(max(h - size, 0))
    if xs[-1] + size < w:
        xs.append(max(w - size, 0))
    return [(y, x) for y in ys for x in xs]


def extract_tiles(img: np.ndarray, boxes: list, size: int = 512, stride: int = 448,
                  min_overlap: float = 0.5):
    """Gera (tile, boxes_locais, (y, x)). Mantem apenas caixas com >= min_overlap
    da area dentro do tile -- caixa cortada pela metade e rotulo ruim."""
    out = []
    for (y, x) in tile_positions(*img.shape[:2], size=size, stride=stride):
        tile = img[y:y + size, x:x + size]
        if tile.shape[0] != size or tile.shape[1] != size:
            pad = np.zeros((size, size), dtype=img.dtype)
            pad[:tile.shape[0], :tile.shape[1]] = tile
            tile = pad
        local = []
        for b in boxes:
            x1, y1, x2, y2 = b[:4]
            area = max((x2 - x1) * (y2 - y1), 1e-6)
            ix1, iy1 = max(x1, x), max(y1, y)
            ix2, iy2 = min(x2, x + size), min(y2, y + size)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / area >= min_overlap:
                local.append([ix1 - x, iy1 - y, ix2 - x, iy2 - y] + list(b[4:]))
        out.append((tile, local, (y, x)))
    return out


def merge_tile_detections(dets, iou_thr: float = 0.3):
    """Traz as deteccoes dos tiles para as coordenadas da imagem e aplica NMS global."""
    from ..eval.froc import iou as _iou

    globals_ = [
        {"bbox": [d["bbox"][0] + x, d["bbox"][1] + y, d["bbox"][2] + x, d["bbox"][3] + y],
         "score": d["score"], "category": d.get("category", "suspicious_calcification")}
        for d, (y, x) in dets
    ]
    globals_.sort(key=lambda d: -d["score"])
    keep = []
    for d in globals_:
        if all(_iou(d["bbox"], k["bbox"]) < iou_thr for k in keep):
            keep.append(d)
    return keep
