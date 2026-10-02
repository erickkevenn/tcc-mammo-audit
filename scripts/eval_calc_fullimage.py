"""Avaliacao do detector de microcalcificacoes na MAMOGRAFIA INTEIRA (S10).

O mAP que o YOLO mostra no treino e medido em recortes escolhidos (1 com
calcificacao para 3 sem). Aqui cada imagem da validacao inteira passa pelo
mesmo preparo do treino (src/preprocess/vindr_calc_tiles.prepare), e percorrida
em recortes de 512 px, as deteccoes voltam para a imagem e as duplicatas da
sobreposicao sao removidas (NMS). Depois: FROC por lesao com acerto pelo CENTRO
da deteccao dentro da caixa do agrupamento, e AUC/sensibilidade por mama, com IC
por bootstrap de exames -- o mesmo src/eval/breast_eval.py do detector de massas.

Todas as imagens da validacao entram (nao so as do treino em recortes), para
que as mamas sem calcificacao tenham a prevalencia real. Leva cerca de 1 h;
as predicoes vao para artifacts/eval/<run>_<split>_preds.csv a cada imagem e o
script CONTINUA de onde parou se for interrompido.

Uso:
    python scripts/eval_calc_fullimage.py --weights runs/detect/artifacts/runs/det_calc_B0/weights/best.pt --limit 20
    python scripts/eval_calc_fullimage.py --weights runs/detect/artifacts/runs/det_calc_B0/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load, paths                                    # noqa: E402
from src.eval.breast_eval import build_units, counts, evaluate        # noqa: E402
from src.preprocess.vindr_calc_tiles import calc_table, prepare, transform_boxes  # noqa: E402
from src.vision.tiles import merge_tile_detections, tile_positions    # noqa: E402

COLS = ["image_id", "x1", "y1", "x2", "y2", "score"]


def tile_image(img8: np.ndarray, mask8: np.ndarray, size: int, stride: int, min_breast: float):
    """Recortes (com preenchimento) que tem pelo menos min_breast de mama."""
    out = []
    for (y, x) in tile_positions(*img8.shape, size=size, stride=stride):
        if mask8[y:y + size, x:x + size].sum() < min_breast * size * size:
            continue
        t = np.zeros((size, size), np.uint8)
        c = img8[y:y + size, x:x + size]
        t[:c.shape[0], :c.shape[1]] = c
        out.append((t, (y, x)))
    return out


def detect_image(predict_fn, tiles, nms_iou: float, max_det: int):
    """predict_fn(lista de recortes) -> lista (por recorte) de [(x1,y1,x2,y2,score)]."""
    if not tiles:
        return []
    per_tile = predict_fn([t for t, _ in tiles])
    dets = [({"bbox": list(b[:4]), "score": float(b[4])}, yx)
            for boxes, (_, yx) in zip(per_tile, tiles) for b in boxes]
    merged = merge_tile_detections(dets, iou_thr=nms_iou)
    return [(tuple(d["bbox"]), d["score"]) for d in merged[:max_det]]


def yolo_predictor(weights: str, imgsz: int, conf: float):
    from ultralytics import YOLO

    model = YOLO(weights)

    def fn(tiles):
        out = []
        for i in range(0, len(tiles), 32):
            batch = [np.repeat(t[:, :, None], 3, axis=2) for t in tiles[i:i + 32]]
            for r in model.predict(source=batch, imgsz=imgsz, conf=conf, verbose=False):
                out.append([(*b, s) for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())])
        return out
    return fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--arm", default="B0")
    ap.add_argument("--um", type=float, default=100.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=448)
    ap.add_argument("--min-breast", type=float, default=0.02, help="pula recortes quase sem mama")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--max-det", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="teste rapido: N exames")
    args = ap.parse_args()
    if args.split == "test" and not args.confirm_test:
        raise SystemExit("O teste roda uma vez, com o modelo congelado. Use --confirm-test se for isso.")

    cfg = paths()
    root = Path(cfg["root"]) / cfg["vindr"]["dir"]
    tab = calc_table(pd.read_csv(root / cfg["vindr"]["finding_csv"]))
    tab = tab[tab.split == args.split]
    if args.limit:
        keep = list(tab[tab.role == "pos"].study_id.unique()[: args.limit // 2]) + \
               list(tab[tab.role != "pos"].study_id.unique()[: args.limit - args.limit // 2])
        tab = tab[tab.study_id.isin(keep)]

    run = Path(args.weights).resolve().parents[1].name
    out_dir = Path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_amostra{args.limit}" if args.limit else ""
    pred_csv = out_dir / f"{run}_{args.split}{tag}_preds.csv"
    gt_csv = out_dir / f"{run}_{args.split}{tag}_gts.csv"
    done = set(pd.read_csv(pred_csv).image_id) if pred_csv.exists() else set()
    gts_rows = pd.read_csv(gt_csv).to_dict("records") if gt_csv.exists() else []
    done &= {r["image_id"] for r in gts_rows}   # so conta como feita se tem predicao E gabarito

    from tqdm import tqdm
    pre = load("preprocess.yaml")
    arm_cfg = pre["arms"][args.arm]
    todo = tab[~tab.image_id.isin(done)]
    print(f"{args.split}: {tab.study_id.nunique()} exames, {len(tab)} imagens; faltam {len(todo)}")
    predict_fn = yolo_predictor(args.weights, args.size, args.conf) if len(todo) else None
    for r in tqdm(todo.itertuples(index=False), total=len(todo), desc="imagens"):
        res = prepare(root / cfg["vindr"]["images"] / r.study_id / f"{r.image_id}.dicom", arm_cfg, pre, args.um)
        if res is None:
            continue
        img8, mask8, info = res
        boxes = transform_boxes(r.boxes, (info["crop_x0"], info["crop_y0"]), info["crop_w"],
                                info["flipped"], info["scale"])
        dets = detect_image(predict_fn, tile_image(img8, mask8, args.size, args.stride, args.min_breast),
                            args.nms_iou, args.max_det)
        rows = [{"image_id": r.image_id, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "score": s}
                for b, s in dets] or [{"image_id": r.image_id, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "score": 0.0}]
        pd.DataFrame(rows, columns=COLS).to_csv(pred_csv, mode="a", header=not pred_csv.exists(), index=False)
        g = [{"image_id": r.image_id, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in boxes] \
            or [{"image_id": r.image_id, "x1": None, "y1": None, "x2": None, "y2": None}]
        pd.DataFrame(g).to_csv(gt_csv, mode="a", header=not gt_csv.exists(), index=False)

    pr = pd.read_csv(pred_csv)
    pr = pr[pr.score > 0]
    gt = pd.read_csv(gt_csv).dropna()
    preds = {i: [((r.x1, r.y1, r.x2, r.y2), r.score) for r in g.itertuples(index=False)]
             for i, g in pr.groupby("image_id")}
    gts = {i: [(r.x1, r.y1, r.x2, r.y2) for r in g.itertuples(index=False)] for i, g in gt.groupby("image_id")}
    units = build_units(tab[["image_id", "study_id", "laterality"]], preds, gts)
    print(f"\n{args.split} | microcalcificacao | acerto: centro dentro da caixa | {counts(units)}")
    tabm = evaluate(units, n_boot=args.n_boot, criterion="center")
    print("\n" + tabm.round(3).to_string())
    res = {"run": run, "split": args.split, "criterion": "center", "counts": counts(units),
           "n_boot": args.n_boot, "metrics": tabm.round(4).to_dict(orient="index")}
    out_json = out_dir / f"{run}_{args.split}{tag}_center.json"
    out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nresultado -> {out_json}")


if __name__ == "__main__":
    main()
