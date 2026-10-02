"""Tabela do classificador por mama (BI-RADS + densidade) a partir do VinDr.

Uma linha por IMAGEM, com o rotulo da MAMA (o breast-level do VinDr vem por
imagem, repetido nas vistas CC e MLO da mesma mama).

Splits: os MESMOS do detector (src/data/vindr_splits.py). A validacao sai de
~10% dos estudos do treino oficial pelo mesmo hash; o teste oficial fica
intocado. Assim o auditor e calibrado num unico conjunto de validacao, sem
vazamento entre detector e classificador.

Rotulos:
    birads  : BI-RADS 1..5 -> 0..4   (nao existe BI-RADS 0 no VinDr)
    density : A+B -> 0, C -> 1, D -> 2  (A tem so 50 mamas: fundida com B)
"""
from __future__ import annotations

import pandas as pd

from .vindr_splits import VAL_SALT, hash01

BIRADS_NAMES = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]
DENSITY_NAMES = ["A+B", "C", "D"]
_DENSITY_IDX = {"A": 0, "B": 0, "C": 1, "D": 2}


def cls_table(breast_df: pd.DataFrame, val_frac: float = 0.10) -> pd.DataFrame:
    """breast_df: saida de src.data.vindr.load_breast_level()."""
    df = breast_df.copy()
    df["birads"] = df["breast_birads"].astype(str).str.extract(r"(\d)")[0].astype(int) - 1
    dens = df["breast_density"].astype(str).str.replace("DENSITY ", "", regex=False).str.strip()
    df["density"] = dens.map(_DENSITY_IDX)
    if df["density"].isna().any() or not df["birads"].between(0, 4).all():
        raise ValueError("rotulo de densidade ou BI-RADS fora do esperado no CSV")
    df["density"] = df["density"].astype(int)
    is_val = df["study_id"].map(lambda s: hash01(s, VAL_SALT) < val_frac)
    df["split"] = df["split"].where(~((df["split"] == "training") & is_val), "val")
    df = df.rename(columns={"view_position": "view"})
    return df[["image_id", "study_id", "laterality", "view", "birads", "density", "split"]]


def check_breast_labels(table: pd.DataFrame) -> int:
    """Quantas mamas tem rotulo diferente entre as vistas (deveria ser 0)."""
    g = table.groupby(["study_id", "laterality"])
    return int(((g.birads.nunique() > 1) | (g.density.nunique() > 1)).sum())


def sample_weights(table: pd.DataFrame, power: float = 0.5) -> pd.Series:
    """Peso por imagem para o WeightedRandomSampler: freq(BI-RADS) ** -power.

    power=0.5 (raiz do inverso) sobe as classes raras sem repetir as mesmas
    poucas imagens de BI-RADS 5 a cada lote; power=0 desliga o balanceamento.
    """
    freq = table.birads.map(table.birads.value_counts(normalize=True))
    return (freq ** -power).astype(float)
