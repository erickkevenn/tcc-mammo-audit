"""ETAPA 0 -- triagem de metadados DICOM SEM ler pixels.

Rode isto ANTES de qualquer conversao. A tabela resultante:
  * responde ao item 13 do CLAIM 2024 (protocolo de aquisicao);
  * revela o domain shift entre datasets (fabricante, bit depth, janela);
  * identifica as imagens 'FOR PROCESSING', que DEVEM ser descartadas
    (nao tem janela de apresentacao valida) -- e o que o codigo oficial do
    VinDr faz e quase todo mundo ignora.

Uso:
    python -m src.data.build_manifest --dataset vindr --limit 200
    python -m src.data.build_manifest --dataset vindr            # tudo
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import data_root, dset, paths

TAGS = [
    "PresentationIntentType", "PhotometricInterpretation", "BitsStored", "BitsAllocated",
    "PixelRepresentation", "RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth",
    "VOILUTFunction", "ImageLaterality", "ViewPosition", "PatientOrientation",
    "PixelSpacing", "ImagerPixelSpacing", "Manufacturer", "ManufacturerModelName",
    "PixelPaddingValue", "Rows", "Columns", "PatientAge",
]


def probe(path: Path) -> dict:
    import pydicom

    row: dict = {"path": str(path), "file": path.name}
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as exc:                                     # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    try:
        row["TransferSyntaxUID"] = str(ds.file_meta.TransferSyntaxUID)
    except Exception:                                            # noqa: BLE001
        row["TransferSyntaxUID"] = None
    for tag in TAGS:
        val = getattr(ds, tag, None)
        if val is not None:
            if tag in ("WindowCenter", "WindowWidth"):
                # VM (value multiplicity) varia por imagem -- 1 valor na maioria,
                # N valores em algumas (por isso 'window_vm' abaixo). Se deixar o
                # tipo variar entre float (VM=1) e string de lista (VM>1), o
                # pyarrow tenta inferir a coluna inteira como double e quebra na
                # primeira linha com VM>1. Normalize para string SEMPRE.
                val = str(val)
            elif not isinstance(val, (str, int, float)):
                val = str(val)
        row[tag] = val
    row["has_voi_lut_sequence"] = (0x0028, 0x3010) in ds
    row["window_vm"] = (
        len(ds.WindowCenter) if hasattr(ds, "WindowCenter")
        and hasattr(ds.WindowCenter, "__len__") and not isinstance(ds.WindowCenter, str)
        else (1 if getattr(ds, "WindowCenter", None) is not None else 0)
    )
    return row


def find_dicoms(dataset: str) -> list[Path]:
    if dataset == "vindr":
        base = dset("vindr") / paths()["vindr"]["images"]
        return sorted(base.rglob("*.dicom")) + sorted(base.rglob("*.dcm"))
    if dataset == "cbis":
        return sorted(dset("cbis").rglob("*.dcm"))
    if dataset == "inbreast":
        return sorted((dset("inbreast") / paths()["inbreast"]["dicoms"]).glob("*.dcm"))
    raise ValueError(f"dataset desconhecido: {dataset} (use vindr|cbis|inbreast)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["vindr", "cbis", "inbreast"])
    ap.add_argument("--limit", type=int, default=0, help="0 = todos")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = find_dicoms(args.dataset)
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} arquivos em {data_root()}")

    rows = [probe(f) for f in tqdm(files, desc="metadados")]
    df = pd.DataFrame(rows)

    out = Path(args.out or f"artifacts/manifests/{args.dataset}_manifest.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"\nsalvo em {out}\n")
    for col in ["PresentationIntentType", "PhotometricInterpretation", "BitsStored",
                "Manufacturer", "TransferSyntaxUID", "VOILUTFunction"]:
        if col in df:
            print(f"-- {col}:")
            print(df[col].value_counts(dropna=False).head(10).to_string(), "\n")
    if "PresentationIntentType" in df:
        n = (df["PresentationIntentType"].astype(str).str.upper() == "FOR PROCESSING").sum()
        print(f">>> {n} imagens FOR PROCESSING serao DESCARTADAS.")
    if "error" in df:
        print(f">>> {df['error'].notna().sum()} arquivos falharam na leitura.")


if __name__ == "__main__":
    main()
