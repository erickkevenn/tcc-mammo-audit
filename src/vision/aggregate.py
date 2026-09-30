"""Agregacao imagem -> mama -> exame.

OBRIGATORIA para a auditoria: o laudo descreve por MAMA, nunca por vista.
Alem disso captura parte do beneficio multi-vista quase de graca -- o ganho
publicado da fusao CC/MLO explicita e +0,074 mAP, mas o VinDr NAO fornece
correspondencia de lesao entre CC e MLO, o que torna a supervisao multi-vista
caro/arriscado para um TCC. Declare isso na secao de limitacoes.
"""
from __future__ import annotations
from collections import defaultdict


def aggregate_detections(dets_by_image: dict[str, list[dict]],
                         image_meta: dict[str, dict],
                         mode: str = "max") -> dict[str, list[dict]]:
    """dets_by_image: {image_id: [{'category','score','bbox'}, ...]}
    image_meta:      {image_id: {'exam_id','laterality','view'}}
    Retorna {f'{exam_id}_{laterality}': [deteccoes com a vista anotada]}"""
    out: dict[str, list[dict]] = defaultdict(list)
    for image_id, dets in dets_by_image.items():
        meta = image_meta.get(image_id, {})
        key = f"{meta.get('exam_id')}_{meta.get('laterality')}"
        for d in dets:
            out[key].append({**d, "image_id": image_id, "view": meta.get("view")})
    merged = {}
    for key, dets in out.items():
        best: dict[str, dict] = {}
        for d in dets:
            cat = d["category"]
            if cat not in best or d["score"] > best[cat]["score"]:
                best[cat] = d
        merged[key] = (list(best.values()) if mode == "max" else dets)
    return merged


def breast_score(dets: list[dict], category: str) -> float:
    """Confianca em nivel de mama para uma categoria = max entre CC e MLO."""
    scores = [d["score"] for d in dets if d["category"] == category]
    return max(scores) if scores else 0.0
