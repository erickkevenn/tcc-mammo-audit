"""Recorte da regiao da mama -- o ganho mais barato de todo o pipeline.

Ablacao limpa em VinDr-Mammo (MamT4): sem recorte ROC-AUC 79,6 -> com recorte 84,0;
F1 44,7 -> 56,0. +4,4 pontos de AUC de graca.

Receita (do codigo do 1o lugar do RSNA 2023, roi_extract.extract_roi_otsu):
  clip p95 -> GaussianBlur(5,5) -> Otsu -> dilate(3,3) -> maior contorno -> boundingRect

O clip em p95 ANTES do Otsu e o truque essencial: remove rotulos queimados
('L MLO'), marcadores metalicos e a borda branca do detector, que de outra
forma dominam o limiar.

NAO remova o musculo peitoral: nenhuma solucao top do RSNA 2023 remove, e em
torax a mascara agressiva DEGRADOU (AUROC 0,742 -> 0,696) enquanto o recorte
manteve. Remova apenas para modulo de densidade.
"""
from __future__ import annotations
import numpy as np


def _clip_level(work: np.ndarray, pct: float = 95.0) -> float:
    """Nivel do corte "p95" calculado SO sobre os pixels que nao sao fundo.

    Corrigido em 30/09/2026. O p95 do quadro inteiro so funciona quando a mama
    domina a imagem. Em mama pequena (vista CC ocupando 6-10% do quadro), o p95
    cai DENTRO da mama e o corte apaga o tecido mais denso: o contorno encolhia
    para 1-4%, ficava abaixo do limite de 4% e o recorte era abandonado (as 20
    falhas de 300 da S2 eram todas assim). Mesmo quando o recorte passava, parte
    da regiao densa, onde as massas aparecem, podia sair do contorno.
    """
    fg = work[work > work.min()]
    return float(np.percentile(fg, pct)) if fg.size else float(work.max())


def breast_bbox(img: np.ndarray, area_pct_thres: float = 0.04) -> tuple[int, int, int, int] | None:
    """Retorna (x0, y0, x1, y1) ou None se a deteccao for implausivel."""
    import cv2

    work = img.astype(np.float32).copy()
    upper = _clip_level(work)
    work[work > upper] = work.min()                          # mata implantes e linhas brancas

    work = work - work.min()
    denom = work.max() if work.max() > 0 else 1.0
    work8 = (work / denom * 255).astype(np.uint8)

    work8 = cv2.GaussianBlur(work8, (5, 5), 0)
    _, binary = cv2.threshold(work8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    h, w = img.shape[:2]
    if cv2.contourArea(cnt) / float(h * w) < area_pct_thres:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    return int(x), int(y), int(x + bw), int(y + bh)


def breast_mask(img: np.ndarray) -> np.ndarray:
    """Mascara booleana da mama -- use para os percentis do Winsor."""
    import cv2

    work = img.astype(np.float32).copy()
    upper = _clip_level(work)
    work[work > upper] = work.min()
    work = work - work.min()
    denom = work.max() if work.max() > 0 else 1.0
    work8 = (work / denom * 255).astype(np.uint8)
    _, binary = cv2.threshold(work8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.ones_like(img, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def canonical_flip(img: np.ndarray, laterality: str | None) -> tuple[np.ndarray, bool]:
    """Orientacao canonica: a parede toracica sempre a ESQUERDA da imagem.

    Convencao (estilo NYU): imagens R sao espelhadas horizontalmente.
    Reduz variancia e ajuda a auditoria a comparar CC e MLO da mesma mama.
    """
    if laterality and str(laterality).upper().startswith("R"):
        return np.ascontiguousarray(img[:, ::-1]), True
    return img, False


def resize_and_pad(img: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, dict]:
    """Preserva a razao de aspecto + padding centralizado com zeros.

    Vencedor da comparacao publicada (Sci Rep 2025): preservar aspecto + padding
    supera resize distorcido para quadrado. Guarde os parametros para poder
    mapear as bounding boxes de volta as coordenadas originais.
    """
    import cv2

    h, w = img.shape[:2]
    ratio = min(target_h / h, target_w / w)
    nh, nw = max(1, int(round(h * ratio))), max(1, int(round(w * ratio)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((target_h, target_w), dtype=img.dtype)
    y0, x0 = (target_h - nh) // 2, (target_w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = resized
    return out, {"ratio": ratio, "pad_x": x0, "pad_y": y0, "new_h": nh, "new_w": nw}


def map_bbox(bbox, crop_offset, transform) -> list[float]:
    """Mapeia (x1,y1,x2,y2) do original para a imagem final. Erre aqui e todo o
    seu mAP fica errado sem dar erro nenhum -- teste com uma caixa conhecida."""
    x1, y1, x2, y2 = bbox
    cx, cy = crop_offset
    r, px, py = transform["ratio"], transform["pad_x"], transform["pad_y"]
    return [(x1 - cx) * r + px, (y1 - cy) * r + py, (x2 - cx) * r + px, (y2 - cy) * r + py]
