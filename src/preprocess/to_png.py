"""ETAPA S2 -- conversor DICOM -> PNG16 (treino) + PNG8 (auditoria) com ROI da mama.

Ao contrario de to_yolo.py (especifico do VinDr + rotulos YOLO), este script
NAO depende de anotacao nenhuma: roda sobre qualquer DICOM de vindr/cbis/inbreast
e serve tanto para a inspecao visual da S2 do cronograma quanto para gerar os
PNG16 que o classificador BI-RADS vai consumir mais adiante.

Ordem do pipeline (obrigatoria -- ver src/preprocess/dicom_io.py):
    Stored Values -> Modality LUT -> ROI da mama (no array cru, ANTES da janela,
    igual ao to_yolo.py) -> intensidade (janela ou Winsor) -> polaridade
    (MONOCHROME1 por ULTIMO) -> CLAHE (se o braco pedir) -> flip canonico ->
    resize+pad.

Uso:
    python -m src.preprocess.to_png --dataset vindr --arm B0 --limit 300
    python -m src.preprocess.to_png --dataset vindr --arm B0              # tudo

Saida (em <out>/):
    png16/<stem>.png   -- 16 bits, mantem a faixa dinamica para treino
    png8/<stem>.png    -- 8 bits, para leitura humana / auditoria
    report.csv         -- 1 linha por imagem: crop_area_pct, bg_mean,
                           polarity_suspect, flipped, erro/skip

Criterio de aceite da S2 (secao 12 do plano, tabela de cronograma):
    0 recortes com area < 4% do quadro; nenhuma imagem invertida (checagem de
    MONOCHROME1); PNG8 legivel. As duas primeiras condicoes sao verificadas
    automaticamente no resumo impresso ao final; a terceira exige inspecao
    visual manual de ~30 PNG8 (o script so consegue provar que o FUNDO ficou
    escuro, nao que o PNG "parece bom" para um humano).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import load
from ..data.build_manifest import find_dicoms
from .breast_roi import breast_bbox, breast_mask, canonical_flip, resize_and_pad
from .dicom_io import apply_windowing, finalize_polarity, read_dicom, winsor_scale
from .roi_input import for_roi


def process_one(dcm_path: Path, arm_cfg: dict, pre: dict) -> dict:
    """Roda o pipeline completo numa imagem; devolve PNG16 + PNG8 + metadados.

    Sempre devolve um dict (nunca None): imagens FOR PROCESSING vem com
    row['skipped'] preenchido e sem 'img16'/'img8'.
    """
    row: dict = {"file": dcm_path.name}
    arr, meta = read_dicom(dcm_path)
    row.update({
        "laterality": meta["laterality"], "view": meta["view"],
        "photometric": meta["photometric"],
    })

    if pre.get("drop_for_processing") and meta["presentation_intent"].upper() == "FOR PROCESSING":
        row["skipped"] = "FOR_PROCESSING"
        return row

    h0, w0 = arr.shape[:2]
    # A segmentacao precisa da mama CLARA sobre fundo escuro; em MONOCHROME1 e o
    # contrario, e o Otsu pegava o fundo (bbox = quadro inteiro). Ver roi_input.py.
    roi_arr = for_roi(arr, meta["photometric"])
    mask = breast_mask(roi_arr)
    crop_offset = (0, 0)
    area_pct = 1.0
    if arm_cfg["crop_breast"]:
        bb = breast_bbox(roi_arr)
        if bb is not None:
            x0, y0, x1, y1 = bb
            area_pct = ((x1 - x0) * (y1 - y0)) / float(h0 * w0)
            arr, mask = arr[y0:y1, x0:x1], mask[y0:y1, x0:x1]
            crop_offset = (x0, y0)
        else:
            area_pct = 0.0
    row["crop_area_pct"] = round(area_pct, 4)
    row["crop_offset"] = crop_offset

    def _intensity(y_max: float) -> np.ndarray:
        if arm_cfg["intensity"] == "winsor":
            return winsor_scale(arr, mask, pre["winsor"]["low"], pre["winsor"]["high"], y_max)
        if meta["wc"] is not None and meta["ww"]:
            return apply_windowing(arr, meta["wc"], meta["ww"], meta["voi_fn"], 0, y_max)
        return winsor_scale(arr, mask, 1.0, 99.0, y_max)

    img16 = np.clip(finalize_polarity(_intensity(65535.0), meta["photometric"], 65535.0),
                     0, 65535).astype(np.uint16)
    img8 = np.clip(finalize_polarity(_intensity(255.0), meta["photometric"], 255.0),
                    0, 255).astype(np.uint8)

    # Checagem de polaridade (criterio S2): com a inversao correta o fundo (fora
    # da mama) tem que ficar ESCURO. Mede ANTES do resize+pad -- o padding com
    # zero mascararia um fundo que tivesse ficado claro por engano.
    p = max(1, min(img8.shape) // 20)
    corners = [img8[:p, :p], img8[:p, -p:], img8[-p:, :p], img8[-p:, -p:]]
    bg_mean = float(np.mean([c.mean() for c in corners]))
    row["bg_mean"] = round(bg_mean, 1)
    row["polarity_suspect"] = bg_mean > 80.0

    if arm_cfg.get("clahe"):
        import cv2
        img8 = cv2.createCLAHE(clipLimit=float(arm_cfg["clahe"]["clip"]),
                                tileGridSize=tuple(arm_cfg["clahe"]["tile"])).apply(img8)

    flipped = False
    if pre.get("canonical_laterality"):
        img8, flipped = canonical_flip(img8, meta["laterality"])
        img16, _ = canonical_flip(img16, meta["laterality"])
    row["flipped"] = flipped

    th, tw = pre["mass_size"]
    img8, _ = resize_and_pad(img8, th, tw)
    img16, _ = resize_and_pad(img16, th, tw)

    row["img8"], row["img16"] = img8, img16
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["vindr", "cbis", "inbreast"])
    ap.add_argument("--arm", default=None, help="B0..B4 (default: preprocess.yaml)")
    ap.add_argument("--limit", type=int, default=0, help="0 = todos")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2

    pre = load("preprocess.yaml")
    arm = args.arm or pre["default_arm"]
    arm_cfg = pre["arms"][arm]

    files = find_dicoms(args.dataset)
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} arquivos, braco {arm}")

    out_root = Path(args.out or f"artifacts/png/{args.dataset}_{arm}")
    (out_root / "png16").mkdir(parents=True, exist_ok=True)
    (out_root / "png8").mkdir(parents=True, exist_ok=True)

    rows = []
    for f in tqdm(files, desc="convertendo"):
        try:
            row = process_one(f, arm_cfg, pre)
        except Exception as exc:                                     # noqa: BLE001
            rows.append({"file": f.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if row.get("skipped"):
            rows.append(row)
            continue
        stem = f.stem
        img16, img8 = row.pop("img16"), row.pop("img8")
        if pre.get("png16_for_training", True):
            cv2.imwrite(str(out_root / "png16" / f"{stem}.png"), img16)
        if pre.get("png8_for_audit", True):
            cv2.imwrite(str(out_root / "png8" / f"{stem}.png"), img8)
        rows.append(row)

    df = pd.DataFrame(rows)
    report = out_root / "report.csv"
    df.to_csv(report, index=False)

    n = len(df)
    n_small = int((df["crop_area_pct"] < 0.04).sum()) if "crop_area_pct" in df else 0
    n_suspect = int(df["polarity_suspect"].fillna(False).sum()) if "polarity_suspect" in df else 0
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    n_skip = int(df["skipped"].notna().sum()) if "skipped" in df else 0

    print(f"\n{n} imagens processadas -> {out_root}")
    print(f">>> {n_small} recortes com area < 4% do quadro (criterio S2: deve ser 0).")
    print(f">>> {n_suspect} imagens com fundo suspeito de polaridade errada (criterio S2: deve ser 0).")
    print(f">>> {n_err} erros de leitura, {n_skip} imagens FOR PROCESSING descartadas.")
    print(f"relatorio completo em {report}")
    print("Proximo passo manual da S2: abra ~30 PNG8 em png8/ e confira visualmente "
          "se a mama esta legivel e o fundo esta preto.")


if __name__ == "__main__":
    main()