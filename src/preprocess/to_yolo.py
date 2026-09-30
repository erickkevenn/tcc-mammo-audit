"""Conversao VinDr -> dataset YOLO, por braco de ablacao (B0..B4).

Uso:
    python -m src.preprocess.to_yolo --arm B0 --limit 5       # amostra: 5 positivas + 5 normais por split
    python -m src.preprocess.to_yolo --arm B0                 # tudo

Faz, por imagem: leitura DICOM -> Modality LUT -> mascara/bbox da mama ->
crop -> intensidade (janela ou Winsor) -> CLAHE (se o braco pedir) ->
flip canonico -> resize+pad -> PNG + rotulo YOLO com as caixas MAPEADAS.

Splits (desde 30/09/2026, ver src/data/vindr_splits.py):
    training : treino oficial menos a validacao, positivas + uma fracao de normais
    val      : ~10% dos ESTUDOS do treino oficial, positivas + normais
    test     : teste oficial intocado, positivas + normais
Imagem normal sai com arquivo de rotulo vazio (background para o YOLO).

Saida padrao: <cache>/yolo_<arm>_v2 (o cache antigo yolo_<arm>, com val=teste
e sem normais, fica onde esta para reproduzir o M1).

O mapeamento de caixa e o ponto onde tudo silenciosamente da errado: rode
tests/test_bbox_mapping.py antes de treinar qualquer coisa.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import dset, load, paths
from ..data.vindr import load_findings
from ..data.vindr_splits import image_table, plan_images, summary
from .breast_roi import breast_bbox, breast_mask, canonical_flip, map_bbox, resize_and_pad
from .dicom_io import apply_windowing, finalize_polarity, read_dicom, winsor_scale
from .roi_input import for_roi


def apply_clahe(img8: np.ndarray, clip: float, tile) -> np.ndarray:
    import cv2
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(tile)).apply(img8)


def process_one(dcm_path: Path, boxes, arm_cfg: dict, pre: dict):
    arr, meta = read_dicom(dcm_path)
    if pre.get("drop_for_processing") and meta["presentation_intent"].upper() == "FOR PROCESSING":
        return None

    # Mesma correcao do to_png: em MONOCHROME1 a mama e escura e o Otsu pegava o
    # fundo, devolvendo o quadro inteiro como ROI. Ver roi_input.py.
    roi_arr = for_roi(arr, meta["photometric"])
    mask = breast_mask(roi_arr)
    crop_offset = (0, 0)
    if arm_cfg["crop_breast"]:
        bb = breast_bbox(roi_arr)
        if bb is not None:
            x0, y0, x1, y1 = bb
            arr, mask = arr[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            crop_offset = (x0, y0)

    if arm_cfg["intensity"] == "winsor":
        img = winsor_scale(arr, mask, pre["winsor"]["low"], pre["winsor"]["high"], 255.0)
    elif meta["wc"] is not None and meta["ww"]:
        img = apply_windowing(arr, meta["wc"], meta["ww"], meta["voi_fn"], 0, 255)
    else:
        img = winsor_scale(arr, mask, 1.0, 99.0, 255.0)

    img = finalize_polarity(img, meta["photometric"], 255.0)
    img8 = np.clip(img, 0, 255).astype(np.uint8)

    if arm_cfg.get("clahe"):
        img8 = apply_clahe(img8, float(arm_cfg["clahe"]["clip"]), arm_cfg["clahe"]["tile"])

    flipped = False
    if pre.get("canonical_laterality"):
        img8, flipped = canonical_flip(img8, meta["laterality"])

    h, w = img8.shape[:2]
    th, tw = pre["mass_size"]
    out, tf = resize_and_pad(img8, th, tw)

    mapped = []
    for (x1, y1, x2, y2, cls) in boxes:
        if flipped:
            ox1 = crop_offset[0]
            width_cropped = w
            x1n = ox1 + (width_cropped - (x2 - ox1))
            x2n = ox1 + (width_cropped - (x1 - ox1))
            x1, x2 = x1n, x2n
        bx = map_bbox((x1, y1, x2, y2), crop_offset, tf)
        cx, cy = (bx[0] + bx[2]) / 2 / tw, (bx[1] + bx[3]) / 2 / th
        bw, bh = (bx[2] - bx[0]) / tw, (bx[3] - bx[1]) / th
        if 0 <= cx <= 1 and 0 <= cy <= 1 and bw > 0 and bh > 0:
            mapped.append((cls, cx, cy, bw, bh))
    return out, mapped, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None, help="B0..B4 (default: preprocess.yaml)")
    ap.add_argument("--split", default=None, choices=["training", "val", "test"])
    ap.add_argument("--limit", type=int, default=0,
                    help="amostra: ate N positivas e N normais POR SPLIT (0 = todas)")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="fracao dos estudos do treino oficial usada como validacao")
    ap.add_argument("--train-neg-ratio", type=float, default=0.25,
                    help="imagens normais no treino por imagem positiva")
    ap.add_argument("--eval-neg-frac", type=float, default=1.0,
                    help="fracao das normais mantida em val/teste (1.0 no numero final)")
    ap.add_argument("--dry-run", action="store_true", help="so mostra o plano, nao converte")
    ap.add_argument("--overwrite", action="store_true",
                    help="apaga images/ e labels/ da pasta de saida antes de escrever")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2

    pre = load("preprocess.yaml")
    arm = args.arm or pre["default_arm"]
    arm_cfg = pre["arms"][arm]
    classes = load("detector_mass.yaml")["classes"]

    img = image_table(load_findings(), classes)
    n_out = int((img.role == "out").sum())
    plan = plan_images(img, args.val_frac, args.train_neg_ratio, args.eval_neg_frac)
    if args.split:
        plan = plan[plan.split == args.split]
    if args.limit:
        plan = plan.groupby(["split", "role"], group_keys=False).head(args.limit)

    print(f"plano ({n_out} imagens so com classes fora do detector ficaram de fora):")
    print(summary(plan).to_string())
    if args.dry_run:
        return

    out_root = Path(args.out or f"{paths()['out']['cache']}/yolo_{arm}_v2")
    existing = out_root / "images"
    if existing.exists() and any(existing.rglob("*.png")):
        if not args.overwrite:
            raise SystemExit(f"{out_root} ja tem imagens. Use --overwrite para refazer "
                             f"(imagens velhas misturariam os splits).")
        shutil.rmtree(out_root / "images", ignore_errors=True)
        shutil.rmtree(out_root / "labels", ignore_errors=True)

    root = dset("vindr")
    n_ok = n_skip = 0
    records = []
    for r in tqdm(plan.itertuples(index=False), total=len(plan), desc=f"braco {arm}"):
        dcm = root / paths()["vindr"]["images"] / r.study_id / f"{r.image_id}.dicom"
        if not dcm.exists():
            dcm = dcm.with_suffix(".dcm")
        if not dcm.exists():
            n_skip += 1
            continue
        res = process_one(dcm, r.boxes, arm_cfg, pre)
        if res is None:
            n_skip += 1
            continue
        out_img, mapped, _ = res
        (out_root / "images" / r.split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / r.split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "images" / r.split / f"{r.image_id}.png"), out_img)
        with open(out_root / "labels" / r.split / f"{r.image_id}.txt", "w") as fh:
            for cls, cx, cy, bw, bh in mapped:
                fh.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        records.append({"image_id": r.image_id, "study_id": r.study_id, "split": r.split,
                        "role": r.role, "n_boxes": len(mapped)})
        n_ok += 1

    import pandas as pd
    pd.DataFrame(records).to_csv(out_root / "split_manifest.csv", index=False)
    yaml_txt = (f"path: {out_root}\ntrain: images/training\nval: images/val\ntest: images/test\n"
                f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
    (out_root / "data.yaml").write_text(yaml_txt, encoding="utf-8")
    print(f"\n{n_ok} imagens escritas, {n_skip} ignoradas -> {out_root}")


if __name__ == "__main__":
    main()