"""Desenha os rotulos YOLO sobre as PNGs de um cache, para conferencia visual.

A caixa vinda de mascara (CBIS-DDSM) e o ponto onde o erro passa silencioso: o
treino roda, a perda cai e o detector aprende a posicao errada. A unica forma
barata de pegar isso e olhar.

Uso:
    python scripts/preview_yolo_labels.py --cache <cache>/yolo_cbis_mass_B0 -n 8
    python scripts/preview_yolo_labels.py --cache <cache>/yolo_B0 --split test -n 8

Saida: <cache>/preview/<nome>.png com a caixa desenhada e o nome da classe,
mais um resumo da area relativa de cada caixa (fracao do quadro).
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path

import cv2
import yaml


def load_names(cache: Path) -> dict[int, str]:
    data = yaml.safe_load((cache / "data.yaml").read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="pasta do cache YOLO")
    ap.add_argument("--split", default="training")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.cache)
    names = load_names(cache)
    img_dir = cache / "images" / args.split
    lbl_dir = cache / "labels" / args.split
    out_dir = cache / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        raise SystemExit(f"nenhuma imagem em {img_dir}")
    random.Random(args.seed).shuffle(imgs)
    imgs = imgs[: args.n]

    print(f"{len(imgs)} imagens de {img_dir}\n")
    for p in imgs:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        lbl = lbl_dir / f"{p.stem}.txt"
        linhas = lbl.read_text().strip().splitlines() if lbl.exists() else []
        areas = []
        for ln in linhas:
            cls, cx, cy, bw, bh = ln.split()
            cls, cx, cy, bw, bh = int(cls), float(cx), float(cy), float(bw), float(bh)
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis, names.get(cls, str(cls)), (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            areas.append(bw * bh)
        cv2.imwrite(str(out_dir / p.name), vis)
        area_txt = ", ".join(f"{a * 100:.1f}%" for a in areas) or "sem caixa"
        flag = ""
        if any(a > 0.5 for a in areas):
            flag = "  <-- caixa ocupa mais de metade do quadro, suspeito"
        if not areas:
            flag = "  <-- rotulo vazio"
        print(f"{p.name}: {len(areas)} caixa(s), area {area_txt}{flag}")

    print(f"\nimagens anotadas em {out_dir}")
    print("Olhe uma por uma: a caixa tem que cair sobre a lesao, dentro da mama, "
          "e nao no meio do fundo preto nem espelhada para o outro lado.")


if __name__ == "__main__":
    main()