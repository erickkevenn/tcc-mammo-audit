"""Conversao VinDr -> dataset YOLO, por braco de ablacao (B0..B4).

Uso:
    python -m src.preprocess.to_yolo --arm B0 --limit 300 --split training
    python -m src.preprocess.to_yolo --arm B0                 # tudo

Faz, por imagem: leitura DICOM -> Modality LUT -> mascara/bbox da mama ->
crop -> intensidade (janela ou Winsor) -> CLAHE (se o braco pedir) ->
flip canonico -> resize+pad -> PNG + rotulo YOLO com as caixas MAPEADAS.

O mapeamento de caixa e o ponto onde tudo silenciosamente da errado: rode
tests/test_bbox_mapping.py antes de treinar qualquer coisa.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import dset, load, paths
from ..data.vindr import load_findings
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
    ap.add_argument("--split", default=None, choices=["training", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2

    pre = load("preprocess.yaml")
    arm = args.arm or pre["default_arm"]
    arm_cfg = pre["arms"][arm]
    classes = load("detector_mass.yaml")["classes"]
    cls_idx = {c: i for i, c in enumerate(classes)}

    df = load_findings()
    if args.split:
        df = df[df.split == args.split]
    df = df[df.has_box]

    root = dset("vindr")
    out_root = Path(args.out or f"{paths()['out']['cache']}/yolo_{arm}")
    grouped = list(df.groupby(["study_id", "image_id"]))
    if args.limit:
        grouped = grouped[: args.limit]

    n_ok = n_skip = 0
    for (study_id, image_id), g in tqdm(grouped, desc=f"braco {arm}"):
        boxes = []
        for _, r in g.iterrows():
            for cat in r["categories_mapped"]:
                if cat in cls_idx:
                    boxes.append((r.xmin, r.ymin, r.xmax, r.ymax, cls_idx[cat]))
        if not boxes:
            continue
        dcm = root / paths()["vindr"]["images"] / study_id / f"{image_id}.dicom"
        if not dcm.exists():
            dcm = dcm.with_suffix(".dcm")
        if not dcm.exists():
            n_skip += 1
            continue
        res = process_one(dcm, boxes, arm_cfg, pre)
        if res is None:
            n_skip += 1
            continue
        img, mapped, _ = res
        split = str(g.iloc[0]["split"])
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root / "images" / split / f"{image_id}.png"), img)
        with open(out_root / "labels" / split / f"{image_id}.txt", "w") as fh:
            for cls, cx, cy, bw, bh in mapped:
                fh.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        n_ok += 1

    yaml_txt = (f"path: {out_root}\ntrain: images/training\nval: images/test\n"
                f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
    (out_root / "data.yaml").write_text(yaml_txt, encoding="utf-8")
    print(f"\n{n_ok} imagens escritas, {n_skip} ignoradas -> {out_root}")


if __name__ == "__main__":
    main()