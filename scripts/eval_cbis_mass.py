"""Teste EXTERNO do detector de massas congelado no CBIS-DDSM (S15).

O CBIS-DDSM e filme digitalizado (nao FFDM) e so tem imagens COM lesao: nao ha
mamas normais, entao a AUC por mama nao existe. Reporta a FROC por lesao (IoU >= 0,5
e centro na caixa), com FP por imagem contados so em imagens com massa.

Usa o cache yolo_cbis_mass_B0_v3 (mesmo preparo do VinDr, orientacao pelo conteudo).
Roda uma vez, com o modelo congelado: exige --confirm-test. As predicoes ficam em
artifacts/eval/<run>_cbis_test_preds.csv e nao sao refeitas se ja existirem.

Uso:  python scripts/eval_cbis_mass.py --confirm-test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import paths                                          # noqa: E402
from src.eval.breast_eval import build_units, counts, evaluate, yolo_to_xyxy  # noqa: E402

MASS_W = "runs/detect/artifacts/runs/det_mass_B0_v2_neg1/weights/best.pt"
SERIES_RE = re.compile(r"_(P_\d+)_(LEFT|RIGHT)_")


def image_table(img_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(img_dir.glob("*.png")):
        m = SERIES_RE.search(p.stem)
        if m:
            rows.append({"image_id": p.stem, "study_id": m[1], "laterality": "L" if m[2] == "LEFT" else "R"})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=MASS_W)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    if not args.confirm_test:
        raise SystemExit("Teste externo: roda uma vez, com o modelo congelado. Use --confirm-test.")
    cache = Path(args.cache or f"{paths()['out']['cache']}/yolo_cbis_mass_B0_v3")
    img_dir, lbl_dir = cache / "images" / "test", cache / "labels" / "test"
    tab = image_table(img_dir)
    if tab.empty:
        raise SystemExit(f"nenhuma imagem em {img_dir}")

    run = Path(args.weights).resolve().parents[1].name
    out_dir = Path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / f"{run}_cbis_test_preds.csv"
    if pred_csv.exists():
        pr = pd.read_csv(pred_csv)
        print(f"usando predicoes salvas: {pred_csv}")
    else:
        from scripts.eval_froc import predict
        pr = predict(args.weights, [img_dir / f"{i}.png" for i in tab.image_id], [0], 1024, 0.001, 20)
        pr.to_csv(pred_csv, index=False)
    pr = pr[pr.cls == 0]
    preds = {i: [((r.x1, r.y1, r.x2, r.y2), r.score) for r in g.itertuples(index=False)]
             for i, g in pr.groupby("image_id")}

    from PIL import Image
    gts = {}
    for iid in tab.image_id:
        w, h = Image.open(img_dir / f"{iid}.png").size
        f = lbl_dir / f"{iid}.txt"
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()] if f.exists() else []
        gts[iid] = [b for c, b in (yolo_to_xyxy(ln, w, h) for ln in lines) if c == 0]
    units = build_units(tab, preds, gts)
    print(f"CBIS-DDSM teste | {counts(units)} | todas as imagens tem massa (sem mamas normais)")
    res = {"run": run, "base": "cbis_ddsm_test", "counts": counts(units), "n_boot": args.n_boot}
    for crit, iou in (("iou", 0.5), ("center", 0.5)):
        t = evaluate(units, n_boot=args.n_boot, criterion=crit, iou_thr=iou)
        t = t[[i.startswith("lesao_") for i in t.index]]
        print(f"\ncriterio {crit}{' IoU>=0,5' if crit == 'iou' else ''}:\n" + t.round(3).to_string())
        res[crit] = t.round(4).to_dict(orient="index")
    out = out_dir / f"{run}_cbis_test.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
