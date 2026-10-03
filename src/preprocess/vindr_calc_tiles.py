"""VinDr -> recortes de 512 px a ~100 um/px para o detector de microcalcificacoes (S9).

Por que recortes: a 1024 x 640 uma microcalcificacao vira 1 pixel. Aqui a mama e
reamostrada para ~100 um/px (o VinDr vem a 85 um/px) e dividida em recortes de
512 px com passo 448 (sobreposicao de 64 px).

As caixas do VinDr marcam o AGRUPAMENTO de calcificacoes, nao cada uma: mediana
~190 px, algumas acima de 1.000 px (maiores que o recorte). Regra para levar uma
caixa a um recorte (`local_boxes`): mantem se >= 50% da caixa cai no recorte OU
se a parte que cai ocupa >= 15% do recorte (agrupamento grande, cortado na
borda). Recorte que toca uma caixa sem atingir a regra fica AMBIGUO e nao entra
nem como positivo nem como negativo.

Negativos: recortes sem caixa com >= 30% de mama, vindos (a) das partes vazias
das imagens com calcificacao e (b) de imagens normais (sem nenhuma caixa no
VinDr), sorteadas de forma deterministica. Proporcao padrao 3 negativos por
positivo, em treino e validacao.

Divisao: a MESMA validacao por exame do detector de massas e do classificador
(src/data/vindr_splits.py). O teste so e gerado com --split test --confirm-test.

Saidas em <out>/: images/<split>/*.png, labels/<split>/*.txt (YOLO, 1 classe),
data.yaml, tiles_manifest.csv (um recorte por linha) e image_manifest.csv (as
transformacoes de cada imagem, para levar deteccoes de volta a imagem inteira).

Uso:
    python -m src.preprocess.vindr_calc_tiles --dry-run
    python -m src.preprocess.vindr_calc_tiles --limit 5          # amostra
    python -m src.preprocess.vindr_calc_tiles                    # treino + validacao
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load, paths
from ..data.vindr_splits import NEG_SALT, VAL_SALT, hash01
from ..vision.tiles import tile_positions

CALC = "Suspicious Calcification"
CLASS_NAME = "suspicious_calcification"


# ----------------------------------------------------------------- tabela
def calc_table(findings: pd.DataFrame, val_frac: float = 0.10) -> pd.DataFrame:
    """Uma linha por imagem: caixas de calcificacao, papel (pos/neg/out) e split."""
    f = findings.copy()
    f["has_box"] = f["xmin"].notna()
    f["is_calc"] = f["finding_categories"].astype(str).str.contains(CALC, regex=False) & f["has_box"]
    rows = []
    for iid, g in f.groupby("image_id", sort=False):
        r0 = g.iloc[0]
        boxes = [(float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax))
                 for r in g[g.is_calc].itertuples()]
        if boxes:
            role = "pos"
        elif not g.has_box.any():
            role = "neg"
        else:
            role = "out"            # tem caixa de outro achado: nao e normal limpa
        split = r0["split"]
        if split == "training" and hash01(r0["study_id"], VAL_SALT) < val_frac:
            split = "val"
        rows.append({"image_id": iid, "study_id": r0["study_id"], "laterality": r0["laterality"],
                     "split": split, "role": role, "boxes": boxes})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- geometria
def transform_boxes(boxes, crop_xy, crop_w, flipped, scale):
    """Leva caixas do DICOM original para a imagem recortada/espelhada/reamostrada."""
    x0, y0 = crop_xy
    out = []
    for (x1, y1, x2, y2) in boxes:
        x1, x2, y1, y2 = x1 - x0, x2 - x0, y1 - y0, y2 - y0
        if flipped:
            x1, x2 = crop_w - x2, crop_w - x1
        out.append((x1 * scale, y1 * scale, x2 * scale, y2 * scale))
    return out


def local_boxes(boxes, y, x, size, keep_frac=0.5, tile_frac=0.15, min_side=6.0):
    """(caixas mantidas no recorte, recorte_ambiguo)."""
    kept, ambiguous = [], False
    for (x1, y1, x2, y2) in boxes:
        ix1, iy1 = max(x1, x), max(y1, y)
        ix2, iy2 = min(x2, x + size), min(y2, y + size)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        if (inter / area >= keep_frac or inter / (size * size) >= tile_frac) \
                and min(ix2 - ix1, iy2 - iy1) >= min_side:
            kept.append((ix1 - x, iy1 - y, ix2 - x, iy2 - y))
        else:
            ambiguous = True
    return kept, ambiguous


def to_yolo_lines(boxes, size):
    return [f"0 {(a + c) / 2 / size:.6f} {(b + d) / 2 / size:.6f} {(c - a) / size:.6f} {(d - b) / size:.6f}"
            for (a, b, c, d) in boxes]


# ----------------------------------------------------------------- imagem
def prepare(dcm_path: Path, arm_cfg: dict, pre: dict, um_per_px: float = 100.0,
            fallback_spacing_mm: float | None = None):
    """DICOM -> (img8 recortada, espelhada e reamostrada; mascara; info). None se descartada.
    Mesma sequencia do to_yolo.process_one, sem o redimensionamento para 1024.

    fallback_spacing_mm: tamanho do pixel usado quando o DICOM nao traz PixelSpacing
    nem ImagerPixelSpacing (caso do INbreast: 0,070 mm, Moreira et al. 2012). Sem ele,
    a imagem nao e reamostrada (escala 1,0), o que foi um defeito nas primeiras
    predicoes do INbreast."""
    import cv2

    from .breast_roi import breast_bbox, breast_mask, canonical_flip, flip_by_content
    from .dicom_io import apply_windowing, finalize_polarity, read_dicom, winsor_scale
    from .roi_input import for_roi

    arr, meta = read_dicom(dcm_path)
    if pre.get("drop_for_processing") and meta["presentation_intent"].upper() == "FOR PROCESSING":
        return None
    roi_arr = for_roi(arr, meta["photometric"])
    mask = breast_mask(roi_arr)
    x0 = y0 = 0
    if arm_cfg["crop_breast"]:
        bb = breast_bbox(roi_arr)
        if bb is not None:
            x0, y0, x1, y1 = bb
            arr, mask = arr[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    if meta["wc"] is not None and meta["ww"]:
        img = apply_windowing(arr, meta["wc"], meta["ww"], meta["voi_fn"], 0, 255)
    else:
        img = winsor_scale(arr, mask, 1.0, 99.0, 255.0)
    img8 = np.clip(finalize_polarity(img, meta["photometric"], 255.0), 0, 255).astype(np.uint8)
    mask8 = (mask > 0).astype(np.uint8)
    crop_w = img8.shape[1]
    if meta.get("laterality"):
        img8, flipped = canonical_flip(img8, meta["laterality"])
    else:
        img8, flipped = flip_by_content(img8)
    if flipped:
        mask8 = np.ascontiguousarray(mask8[:, ::-1])
    sp, sp_src = meta.get("pixel_spacing"), "dicom"
    if not sp and fallback_spacing_mm:
        sp, sp_src = [fallback_spacing_mm], "padrao"
    elif not sp:
        sp_src = "ausente"
    scale = (float(sp[0]) * 1000.0 / um_per_px) if sp else 1.0
    if abs(scale - 1.0) > 1e-3:
        h, w = img8.shape
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        img8 = cv2.resize(img8, (nw, nh), interpolation=cv2.INTER_AREA)
        mask8 = cv2.resize(mask8, (nw, nh), interpolation=cv2.INTER_NEAREST)
    info = {"crop_x0": x0, "crop_y0": y0, "crop_w": crop_w, "flipped": flipped,
            "scale": round(scale, 6), "h": img8.shape[0], "w": img8.shape[1], "spacing_fonte": sp_src}
    return img8, mask8, info


# ----------------------------------------------------------------- principal
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="training,val", help="training,val (padrao) ou test")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--arm", default="B0")
    ap.add_argument("--um", type=float, default=100.0, help="micrometros por pixel apos reamostragem")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=448)
    ap.add_argument("--neg-ratio", type=float, default=3.0, help="recortes negativos por positivo")
    ap.add_argument("--neg-images", default="training=400,val=150,test=400",
                    help="quantas imagens normais sortear por split")
    ap.add_argument("--min-breast", type=float, default=0.30, help="fracao minima de mama no recorte negativo")
    ap.add_argument("--limit", type=int, default=0, help="amostra: N imagens positivas e N normais por split")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    splits = [s.strip() for s in args.split.split(",")]
    if "test" in splits and not args.confirm_test:
        raise SystemExit("O teste so e usado uma vez, com o modelo congelado. Use --confirm-test.")
    neg_n = {k: int(v) for k, v in (p.split("=") for p in args.neg_images.split(","))}

    cfg = paths()
    root = Path(cfg["root"]) / cfg["vindr"]["dir"]
    tab = calc_table(pd.read_csv(root / cfg["vindr"]["finding_csv"]))
    tab = tab[tab.split.isin(splits)]
    pos = tab[tab.role == "pos"]
    neg = tab[tab.role == "neg"].assign(h=lambda d: d.image_id.map(lambda s: hash01(s, NEG_SALT)))
    neg = pd.concat([g.nsmallest(neg_n.get(s, 0), "h") for s, g in neg.groupby("split")])
    if args.limit:
        pos = pos.groupby("split").head(args.limit)
        neg = neg.groupby("split").head(args.limit)
    work = pd.concat([pos, neg.drop(columns="h")], ignore_index=True)

    print(pd.crosstab(work.split, work.role).to_string())
    print(f"caixas de calcificacao: {int(pos.boxes.map(len).sum())}")
    if args.dry_run:
        return

    out = Path(args.out or f"{cfg['out']['cache']}/calc_tiles_{args.arm}"
               + (f"_amostra{args.limit}" if args.limit else ""))
    if out.exists() and any(out.rglob("*.png")) and not args.overwrite:
        raise SystemExit(f"{out} ja tem imagens. Use --overwrite ou outro --out.")
    for s in splits:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    import cv2
    from tqdm import tqdm

    pre = load("preprocess.yaml")
    arm_cfg = pre["arms"][args.arm]
    size, stride = args.size, args.stride
    tiles, images, pools = [], [], {s: [] for s in splits}
    for r in tqdm(work.itertuples(index=False), total=len(work), desc="recortes"):
        dcm = root / cfg["vindr"]["images"] / r.study_id / f"{r.image_id}.dicom"
        try:
            res = prepare(dcm, arm_cfg, pre, args.um)
        except Exception as exc:                                   # noqa: BLE001
            images.append({"image_id": r.image_id, "split": r.split, "role": r.role, "error": str(exc)})
            continue
        if res is None:
            images.append({"image_id": r.image_id, "split": r.split, "role": r.role, "skipped": "FOR_PROCESSING"})
            continue
        img8, mask8, info = res
        boxes = transform_boxes(r.boxes, (info["crop_x0"], info["crop_y0"]), info["crop_w"],
                                info["flipped"], info["scale"])
        images.append({"image_id": r.image_id, "study_id": r.study_id, "split": r.split,
                       "role": r.role, "n_boxes": len(boxes), **info})
        for (y, x) in tile_positions(*img8.shape, size=size, stride=stride):
            kept, amb = local_boxes(boxes, y, x, size)
            tile = np.zeros((size, size), np.uint8)
            crop = img8[y:y + size, x:x + size]
            tile[:crop.shape[0], :crop.shape[1]] = crop
            frac = float(mask8[y:y + size, x:x + size].sum()) / (size * size)
            name = f"{r.image_id}_{y}_{x}"
            row = {"tile": name, "image_id": r.image_id, "study_id": r.study_id, "split": r.split,
                   "y": y, "x": x, "breast_frac": round(frac, 3), "n_boxes": len(kept)}
            if kept:
                cv2.imwrite(str(out / "images" / r.split / f"{name}.png"), tile)
                (out / "labels" / r.split / f"{name}.txt").write_text("\n".join(to_yolo_lines(kept, size)) + "\n")
                tiles.append({**row, "kind": "pos"})
            elif not amb and frac >= args.min_breast:
                pools[r.split].append((hash01(name, NEG_SALT), row, tile))

    # negativos: neg_ratio por positivo, sorteio deterministico pelo hash do nome
    for s in splits:
        n_pos = sum(1 for t in tiles if t["split"] == s and t["kind"] == "pos")
        chosen = sorted(pools[s], key=lambda t: t[0])[: int(round(args.neg_ratio * n_pos))]
        for _, row, tile in chosen:
            cv2.imwrite(str(out / "images" / s / f"{row['tile']}.png"), tile)
            (out / "labels" / s / f"{row['tile']}.txt").write_text("")
            tiles.append({**row, "kind": "neg"})
        print(f"{s}: {n_pos} recortes positivos, {len(chosen)} negativos (de {len(pools[s])} candidatos)")

    pd.DataFrame(tiles).to_csv(out / "tiles_manifest.csv", index=False)
    pd.DataFrame(images).to_csv(out / "image_manifest.csv", index=False)
    lines = [f"path: {out.as_posix()}"]
    for s in splits:
        lines.append(f"{'train' if s == 'training' else s}: images/{s}")
    lines += ["names:", f"  0: {CLASS_NAME}"]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"pronto -> {out}")


if __name__ == "__main__":
    main()
