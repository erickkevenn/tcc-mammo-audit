"""Parser do VinDr-Mammo.

Fatos medidos nos CSVs (19/08/2026), use-os como teste de sanidade:
  finding_annotations.csv : 20.486 linhas; 2.254 com bounding box
  breast-level            : 20.000 linhas
  split                   : training 16.391 / test 4.095
  finding_birads          : SOMENTE 'BI-RADS 3' (893), 4 (995), 5 (241)  -> nunca 1 ou 2
  breast_birads           : 1:13.406  2:4.676  3:930  4:762  5:226   (nao existe BI-RADS 0)
  breast_density          : C:15.292  D:2.700  B:1.908  A:100  (A e degenerada: funda com B)
"""
from __future__ import annotations
import ast
from pathlib import Path
import pandas as pd

from ..config import dset, paths

# Fusao usada pelo detector: classes com N < ~60 caixas viram 'other_suspicious'.
CATEGORY_MAP = {
    "Mass": "mass",
    "Suspicious Calcification": "suspicious_calcification",
    "Architectural Distortion": "architectural_distortion",
    "Focal Asymmetry": "focal_asymmetry",
    "Asymmetry": "asymmetry",
    "Global Asymmetry": "global_asymmetry",
    "Skin Thickening": "other_suspicious",
    "Skin Retraction": "other_suspicious",
    "Nipple Retraction": "other_suspicious",
    "Suspicious Lymph Node": "other_suspicious",
}


def _parse_categories(value) -> list[str]:
    """finding_categories vem como STRING de lista Python: "['Mass']".

    Usar split(',') aqui e um bug classico: quebra em combinacoes como
    "['Suspicious Calcification', 'Mass']" (82 ocorrencias reais).
    """
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    return [str(x).strip() for x in parsed]


def load_findings(root: Path | None = None) -> pd.DataFrame:
    base = root or dset("vindr")
    cfg = paths()["vindr"]
    df = pd.read_csv(base / cfg["finding_csv"])
    df["categories"] = df["finding_categories"].apply(_parse_categories)
    df["categories_mapped"] = df["categories"].apply(
        lambda cs: sorted({CATEGORY_MAP.get(c, "other_suspicious") for c in cs})
    )
    df["has_box"] = df["xmin"].notna()
    df["birads_num"] = (
        df["breast_birads"].astype(str).str.extract(r"(\d)").astype("float")
    )
    df["finding_birads_num"] = (
        df["finding_birads"].astype(str).str.extract(r"(\d)").astype("float")
    )
    df["density"] = df["breast_density"].astype(str).str.replace("DENSITY ", "", regex=False)
    # Densidade A tem apenas 100 linhas (50 mamas): funda A+B para nao criar classe degenerada.
    df["density_merged"] = df["density"].replace({"A": "AB", "B": "AB"})
    df["breast_key"] = df["study_id"] + "_" + df["laterality"]
    return df


def load_breast_level(root: Path | None = None) -> pd.DataFrame:
    base = root or dset("vindr")
    cfg = paths()["vindr"]
    df = pd.read_csv(base / cfg["breast_csv"])
    df["breast_key"] = df["study_id"] + "_" + df["laterality"]
    return df


def image_path(root: Path, study_id: str, image_id: str) -> Path:
    return root / paths()["vindr"]["images"] / study_id / f"{image_id}.dicom"


def sanity_check(df: pd.DataFrame) -> dict:
    """Compare com os valores do docstring. Divergencia = dataset diferente do auditado."""
    return {
        "rows": len(df),
        "rows_with_box": int(df["has_box"].sum()),
        "splits": df["split"].value_counts().to_dict(),
        "finding_birads": df["finding_birads"].value_counts(dropna=True).to_dict(),
        "studies": df["study_id"].nunique(),
        "images": df["image_id"].nunique(),
    }


if __name__ == "__main__":
    import json
    d = load_findings()
    print(json.dumps(sanity_check(d), indent=2, ensure_ascii=False))
