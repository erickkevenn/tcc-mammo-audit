"""Checa a polaridade em uma amostra ESCOLHIDA do manifesto, nao nos primeiros N.

O `to_png --limit N` pega os N primeiros arquivos em ordem alfabetica, o que no
VinDr cai quase todo em um fabricante so. Como MONOCHROME1 esta concentrado em
um fabricante (Planmed), esse recorte nunca exercita a inversao de polaridade.
Aqui a amostra e sorteada a partir do manifesto, filtrando por fabricante ou por
PhotometricInterpretation.

Uso:
    python scripts/check_polarity.py --photometric MONOCHROME1 -n 40
    python scripts/check_polarity.py --manufacturer Planmed -n 40 --save

Criterio: com a polaridade correta, o fundo (cantos, fora da mama) fica ESCURO.
bg_mean alto (> 80 em 0-255) indica imagem invertida.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load
from src.preprocess.to_png import process_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifests/vindr_manifest.parquet")
    ap.add_argument("--photometric", default=None, help="MONOCHROME1 | MONOCHROME2")
    ap.add_argument("--manufacturer", default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true", help="grava os PNG8 para olhar")
    ap.add_argument("--out", default="artifacts/png/polarity_check")
    args = ap.parse_args()

    pre = load("preprocess.yaml")
    arm = args.arm or pre["default_arm"]
    arm_cfg = pre["arms"][arm]

    df = pd.read_parquet(args.manifest)
    if args.photometric:
        df = df[df["PhotometricInterpretation"].astype(str) == args.photometric]
    if args.manufacturer:
        df = df[df["Manufacturer"].astype(str).str.contains(args.manufacturer, case=False, na=False)]
    if df.empty:
        raise SystemExit("nenhuma imagem no manifesto com esse filtro")

    sample = df.sample(min(args.n, len(df)), random_state=args.seed)
    print(f"{len(sample)} imagens sorteadas de {len(df)} candidatas, braco {arm}\n")

    out_dir = Path(args.out)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    linhas = []
    for path in sample["path"]:
        p = Path(path)
        try:
            row = process_one(p, arm_cfg, pre)
        except Exception as exc:                                     # noqa: BLE001
            linhas.append({"file": p.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if row.get("skipped"):
            linhas.append({"file": p.name, "skipped": row["skipped"]})
            continue
        if args.save:
            import cv2
            cv2.imwrite(str(out_dir / f"{p.stem}.png"), row["img8"])
        linhas.append({
            "file": p.name, "photometric": row["photometric"],
            "bg_mean": row["bg_mean"], "polarity_suspect": row["polarity_suspect"],
            "crop_area_pct": row["crop_area_pct"],
        })

    res = pd.DataFrame(linhas)
    print(res.to_string(index=False))
    if "polarity_suspect" in res:
        n_susp = int(res["polarity_suspect"].fillna(False).sum())
        print(f"\n>>> {n_susp} de {len(res)} com fundo suspeito (criterio: 0).")
        print(f">>> bg_mean: mediana {res['bg_mean'].median():.1f}, maximo {res['bg_mean'].max():.1f}")
    if args.save:
        print(f"\nPNG8 em {out_dir} — abra uns 5 e confirme fundo preto e mama legivel.")


if __name__ == "__main__":
    main()