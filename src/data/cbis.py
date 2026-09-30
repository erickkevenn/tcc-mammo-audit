"""Parser do CBIS-DDSM.

ATENCAO -- armadilhas verificadas nos seus arquivos:
 1. Massas usam a coluna 'breast_density' (underscore); calcificacoes usam
    'breast density' (espaco). Normalize antes de concatenar.
 2. As colunas de caminho TROCAM 'cropped image' e 'ROI mask' de forma
    inconsistente entre linhas. NAO confie nelas: classifique por conteudo
    (a mascara e binaria e tem as dimensoes da imagem completa).
 3. 'assessment' contem o valor 0 (incompleto). Decida explicitamente se exclui.
 4. Existem 'subtlety'=0 e 'breast density'=0, fora de escala.
 5. Vazamento cruzado nos splits oficiais: 13 pacientes de mass_train aparecem
    em calc_test e 18 de calc_train em mass_test. Se fundir massa+calc, refaca
    o split por paciente (ver src/data/splits.py).
 6. E FILME DIGITALIZADO, nao FFDM. Use como pre-treino / teste de robustez,
    nunca como treino principal nem avaliacao principal de um sistema FFDM.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from ..config import dset, paths

RENAME = {
    "breast density": "breast_density",
    "left or right breast": "laterality_raw",
    "image view": "view",
    "abnormality id": "abnormality_id",
    "abnormality type": "abnormality_type",
    "mass shape": "mass_shape",
    "mass margins": "mass_margins",
    "calc type": "calc_type",
    "calc distribution": "calc_distribution",
    "image file path": "path_full",
    "cropped image file path": "path_cropped",
    "ROI mask file path": "path_mask",
}


def load_all(root: Path | None = None) -> pd.DataFrame:
    base = (root or dset("cbis")) / paths()["cbis"]["metadata_dir"]
    frames = []
    for name in paths()["cbis"]["csvs"]:
        df = pd.read_csv(base / name)
        df = df.rename(columns=RENAME)
        df["source_csv"] = name
        df["official_split"] = "test" if "test" in name else "train"
        df["kind"] = "mass" if name.startswith("mass") else "calc"
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["laterality"] = out["laterality_raw"].str[0]     # LEFT -> L
    out["series_base"] = (
        out["kind"].str.capitalize()
        + "-"
        + out["official_split"].str.capitalize().replace({"Train": "Training"})
        + "_"
        + out["patient_id"]
        + "_"
        + out["laterality_raw"]
        + "_"
        + out["view"]
    )
    out["series_roi"] = out["series_base"] + "_" + out["abnormality_id"].astype(str)
    return out


def resolve_cropped_and_mask(folder: Path) -> dict:
    """Classifica os DICOMs de uma pasta '..._N' por CONTEUDO, nao por nome.

    Regra: a mascara e binaria (<= 2 valores distintos); o cropped e
    multivalorado. Isso resolve a troca inconsistente das colunas de caminho.
    """
    import numpy as np
    import pydicom

    out = {"mask": None, "cropped": None}
    for dcm in sorted(folder.rglob("*.dcm")):
        arr = pydicom.dcmread(str(dcm)).pixel_array
        if len(np.unique(arr)) <= 2:
            out["mask"] = dcm
        else:
            out["cropped"] = dcm
    return out


def leakage_report(df: pd.DataFrame) -> dict:
    m_tr = set(df[(df.kind == "mass") & (df.official_split == "train")].patient_id)
    m_te = set(df[(df.kind == "mass") & (df.official_split == "test")].patient_id)
    c_tr = set(df[(df.kind == "calc") & (df.official_split == "train")].patient_id)
    c_te = set(df[(df.kind == "calc") & (df.official_split == "test")].patient_id)
    return {
        "mass_train_in_calc_test": len(m_tr & c_te),
        "calc_train_in_mass_test": len(c_tr & m_te),
        "mass_train_in_mass_test": len(m_tr & m_te),
        "calc_train_in_calc_test": len(c_tr & c_te),
        "patients_total": df.patient_id.nunique(),
    }


if __name__ == "__main__":
    import json
    d = load_all()
    print(len(d), "linhas |", d.patient_id.nunique(), "pacientes")
    print(json.dumps(leakage_report(d), indent=2))
