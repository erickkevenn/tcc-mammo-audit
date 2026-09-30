"""FROC por lesao e desempenho por MAMA, com IC 95% bootstrap por exame.

    python scripts/eval_froc.py --weights runs/detect/artifacts/runs/det_mass_B0_v2_neg1/weights/best.pt --cache E:/Erick/TCC/tcc_cache/yolo_B0_v2_neg1

Roda na VALIDACAO por padrao. O teste so roda com --split test --confirm-test,
uma unica vez, com o modelo congelado (M2/S14).

As predicoes ficam salvas em artifacts/eval/<run>_<split>_preds.csv: rodar de
novo (outro criterio, mais reamostragens) nao chama o modelo outra vez.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.vindr import load_findings
from src.eval.breast_eval import build_units, counts, evaluate, yolo_to_xyxy


def predict(weights: str, paths: list[Path], cls_ids: list[int], imgsz: int,
            conf: float, max_det: int) -> pd.DataFrame:
    from ultralytics import YOLO
    from tqdm import tqdm

    model = YOLO(weights)
    rows = []
    for i in tqdm(range(0, len(paths), 32), desc="predicoes"):
        chunk = [str(p) for p in paths[i:i + 32]]
        for r in model.predict(source=chunk, imgsz=imgsz, conf=conf, max_det=max_det,
                               classes=cls_ids, verbose=False, stream=True):
            iid = Path(r.path).stem
            for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                              r.boxes.cls.tolist()):
                rows.append({"image_id": iid, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "score": s, "cls": int(c)})
    return pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "score", "cls"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--classes", default="mass", help="classes avaliadas, separadas por virgula")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--criterion", default="iou", choices=["iou", "center"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--conf", type=float, default=0.005)
    ap.add_argument("--max-det", type=int, default=10)
    ap.add_argument("--repredict", action="store_true", help="ignora as predicoes salvas")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        raise SystemExit("O teste roda uma vez, com o modelo congelado. Use --confirm-test se for isso.")

    from PIL import Image

    cache = Path(args.cache)
    names = yaml.safe_load((cache / "data.yaml").read_text(encoding="utf-8"))["names"]
    wanted = [c.strip() for c in args.classes.split(",")]
    cls_ids = [i for i, n in names.items() if n in wanted]
    if not cls_ids:
        raise SystemExit(f"nenhuma classe {wanted} em {list(names.values())}")

    man = pd.read_csv(cache / "split_manifest.csv")
    man = man[man.split == args.split]
    lat = load_findings().drop_duplicates("image_id").set_index("image_id")["laterality"]
    man = man.assign(laterality=man.image_id.map(lat))
    if man.laterality.isna().any():
        raise SystemExit(f"{man.laterality.isna().sum()} imagens sem lateralidade no CSV do VinDr")

    img_dir = cache / "images" / args.split
    lbl_dir = cache / "labels" / args.split
    paths = [img_dir / f"{i}.png" for i in man.image_id]

    run = Path(args.weights).resolve().parents[1].name
    out_dir = Path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / f"{run}_{args.split}_preds.csv"
    if pred_csv.exists() and not args.repredict:
        pr = pd.read_csv(pred_csv)
        print(f"usando predicoes salvas: {pred_csv}")
    else:
        pr = predict(args.weights, paths, cls_ids, args.imgsz, args.conf, args.max_det)
        pr.to_csv(pred_csv, index=False)
    pr = pr[pr.cls.isin(cls_ids)]

    preds = {iid: [((r.x1, r.y1, r.x2, r.y2), r.score) for r in g.itertuples(index=False)]
             for iid, g in pr.groupby("image_id")}
    gts = {}
    for iid, p in zip(man.image_id, paths):
        w, h = Image.open(p).size
        lines = [ln for ln in (lbl_dir / f"{iid}.txt").read_text().splitlines() if ln.strip()]
        boxes = [yolo_to_xyxy(ln, w, h) for ln in lines]
        gts[iid] = [b for c, b in boxes if c in cls_ids]

    units = build_units(man[["image_id", "study_id", "laterality"]], preds, gts)
    print(f"\n{args.split} | classes {wanted} | criterio {args.criterion}"
          f"{'' if args.criterion == 'center' else f' IoU>={args.iou}'} | {counts(units)}")
    tab = evaluate(units, n_boot=args.n_boot, criterion=args.criterion, iou_thr=args.iou)
    print("\n" + tab.round(3).to_string())
    if tab.loc["lesao_fppi_max", "valor"] < 4:
        print("\naviso: a curva FROC parou antes de 4 FP/imagem; a pAUC esta truncada. "
              "Rode de novo com --repredict --conf 0.001 --max-det 20.")

    res = {"run": run, "split": args.split, "classes": wanted, "criterion": args.criterion,
           "iou": args.iou, "counts": counts(units), "n_boot": args.n_boot,
           "metrics": tab.round(4).to_dict(orient="index")}
    out_json = out_dir / f"{run}_{args.split}_{args.criterion}.json"
    out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresultado salvo em {out_json}")


if __name__ == "__main__":
    main()
