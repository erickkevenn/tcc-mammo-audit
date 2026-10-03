"""Predicoes dos tres modelos CONGELADOS nas 410 imagens do INbreast (preparo da S13).

So gera e salva predicoes: NAO calcula metrica nenhuma. O INbreast e o conjunto de
avaliacao do sistema com laudos reais; os resultados so serao olhados na S14,
com tudo congelado (modelos e limiares em configs/verifier_thresholds.yaml).

  massa           det_mass_B0_v2_neg1 nos PNG8 de 1024 x 640 (png_inbreast_B0_v2)
  microcalcif.    det_calc_B0 em recortes de 512 px a 100 um/px, a partir do DICOM
  classificador   cls_b2_imagenet nos PNG16

Saidas em artifacts/inbreast/: mass_preds.csv, calc_preds.csv, cls_preds.csv.
Cada parte e pulada se o arquivo ja existir (use --overwrite para refazer).

Uso:
    python scripts/predict_inbreast.py --limit 4     # teste rapido
    python scripts/predict_inbreast.py               # tudo (~15 a 20 min)

Ordem: classificador, massa, microcalcificacao (ver o comentario em main).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import dset, load, paths                            # noqa: E402

MASS_W = "runs/detect/artifacts/runs/det_mass_B0_v2_neg1/weights/best.pt"
CALC_W = "runs/detect/artifacts/runs/det_calc_B0/weights/best.pt"
CLS_W = "artifacts/cls_runs/cls_b2_imagenet/best.pt"
NAME_RE = re.compile(r"^(?P<acc>\d+)_(?P<patient>[0-9a-f]+)_MG_(?P<lat>[LR])_(?P<view>[A-Z]+)_ANON$")


def image_table(png_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((png_dir / "png8").glob("*.png")):
        m = NAME_RE.match(p.stem)
        if m:
            rows.append({"image_id": p.stem, "acc": m["acc"], "study_id": m["patient"],
                         "laterality": m["lat"], "view": "MLO" if m["view"] == "ML" else m["view"],
                         "birads": 0, "density": 0})
    return pd.DataFrame(rows)


def predict_mass(tab, png_dir, weights, out_csv):
    from ultralytics import YOLO
    from tqdm import tqdm
    model, rows = YOLO(weights), []
    paths_ = [str(png_dir / "png8" / f"{i}.png") for i in tab.image_id]
    for i in tqdm(range(0, len(paths_), 16), desc="massa"):
        for r in model.predict(source=paths_[i:i + 16], imgsz=1024, conf=0.001, max_det=20,
                               classes=[0], verbose=False):
            for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                rows.append({"image_id": Path(r.path).stem, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": s})
    pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "score"]).to_csv(out_csv, index=False)
    print(f"massa: {len(rows)} deteccoes em {len(tab)} imagens -> {out_csv}")


def predict_calc(tab, weights, out_csv, fallback_mm=None):
    from tqdm import tqdm
    from scripts.eval_calc_fullimage import detect_image, tile_image, yolo_predictor
    from src.preprocess.vindr_calc_tiles import prepare
    pre = load("preprocess.yaml")
    arm = pre["arms"]["B0"]
    dcm_dir = dset("inbreast") / paths()["inbreast"]["dicoms"]
    fn, rows = yolo_predictor(weights, 512, 0.001), []
    for r in tqdm(tab.itertuples(index=False), total=len(tab), desc="microcalcificacao"):
        res = prepare(dcm_dir / f"{r.image_id}.dcm", arm, pre, 100.0, fallback_spacing_mm=fallback_mm)
        if res is None:
            continue
        img8, mask8, info = res
        for b, s in detect_image(fn, tile_image(img8, mask8, 512, 448, 0.02), 0.3, 20):
            rows.append({"image_id": r.image_id, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "score": s,
                         **{k: info[k] for k in ("crop_x0", "crop_y0", "crop_w", "flipped", "scale",
                                                 "spacing_fonte")}})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"microcalcificacao: {len(rows)} deteccoes em {len(tab)} imagens -> {out_csv}")


def predict_cls(tab, png_dir, weights, out_csv):
    if "ultralytics" in sys.modules:
        raise RuntimeError("rode o classificador antes de importar o ultralytics (cv2.imread trocado)")
    import torch
    from torch.utils.data import DataLoader
    from src.vision.train_breast_classifier import BreastClassifier, BreastImages, predict_images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BreastClassifier("tf_efficientnet_b2", pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    t = tab.reset_index(drop=True)
    dl = DataLoader(BreastImages(t, png_dir, 16, train=False), batch_size=8, shuffle=False, num_workers=0)
    x0 = dl.dataset[0][0]
    if tuple(x0.shape) != (1, 1024, 640):
        raise RuntimeError(f"entrada inesperada para o classificador: {tuple(x0.shape)}")
    img = predict_images(model, dl, t, device).drop(columns=["birads", "density"])
    img.to_csv(out_csv, index=False)
    print(f"classificador: {len(img)} imagens -> {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--calc-spacing-mm", type=float, default=None,
                    help="CORRECAO pos-teste: tamanho do pixel quando o DICOM nao traz (INbreast: 0.07). "
                         "Roda so a microcalcificacao e grava calc_preds_esc<um>.csv, sem apagar a original")
    args = ap.parse_args()
    png_dir = Path(args.png_dir or f"{paths()['out']['cache']}/png_inbreast_B0_v2")
    tab = image_table(png_dir)
    if tab.empty:
        raise SystemExit(f"nenhum PNG do INbreast em {png_dir / 'png8'}")
    if args.limit:
        tab = tab.head(args.limit)
    out = Path("artifacts/inbreast")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_amostra{args.limit}" if args.limit else ""
    print(f"{len(tab)} imagens, {tab.study_id.nunique()} pacientes")
    if args.calc_spacing_mm:
        o = out / f"calc_preds_esc{round(args.calc_spacing_mm * 1000)}{tag}.csv"
        if o.exists() and not args.overwrite:
            raise SystemExit(f"{o} ja existe")
        predict_calc(tab, CALC_W, o, args.calc_spacing_mm)
        p = pd.read_csv(o)
        print("escala usada:", p.scale.round(3).value_counts().to_dict(),
              "| fonte do pixel:", p.spacing_fonte.value_counts().to_dict())
        return
    # O classificador roda PRIMEIRO: ao ser importado, o ultralytics troca o
    # cv2.imread por uma versao propria que devolve imagem colorida, e o PNG16
    # de um canal chegava ao classificador com 3 canais.
    for name, fn in (("cls", lambda o: predict_cls(tab, png_dir, CLS_W, o)),
                     ("mass", lambda o: predict_mass(tab, png_dir, MASS_W, o)),
                     ("calc", lambda o: predict_calc(tab, CALC_W, o))):
        o = out / f"{name}_preds{tag}.csv"
        if o.exists() and not args.overwrite:
            print(f"{o} ja existe, pulando")
            continue
        fn(o)
    print("pronto. Nenhuma metrica foi calculada (avaliacao so na S14).")


if __name__ == "__main__":
    main()
