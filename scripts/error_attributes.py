"""Analise de erro por atributo (S15): em que mamas os detectores acertam ou falham
no limiar da regra V1, no teste oficial do VinDr-Mammo. Nada aqui muda resultado.

Nivel de MAMA, como na regra V1: a mama e detectada se a maior pontuacao entre as
suas vistas passa do limiar congelado. Para as mamas com lesao marcada, mostra a
taxa de deteccao por categoria BI-RADS do achado, por densidade e por tamanho da
caixa (em mm, com o tamanho do pixel do DICOM quando o manifesto tem). Para as
mamas de categoria 1 e 2 sem caixa, mostra a taxa de alerta falso por densidade.

Saidas: artifacts/erro/atributos_<massa|calc>.csv e o resumo no terminal.
Uso:  python scripts/error_attributes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("artifacts/erro")
KINDS = {
    "massa": {"term": "Mass", "preds": "artifacts/eval/det_mass_B0_v2_neg1_test_preds.csv",
              "thr": "mass", "bins": [0, 10, 20, np.inf], "labels": ["< 10 mm", "10 a 20 mm", ">= 20 mm"]},
    "calc": {"term": "Suspicious Calcification", "preds": "artifacts/eval/det_calc_B0_test_preds.csv",
             "thr": "calc", "bins": [0, 10, 30, np.inf], "labels": ["< 10 mm", "10 a 30 mm", ">= 30 mm"]},
}


def spacing_table() -> pd.Series:
    """image_id -> tamanho do pixel em mm, a partir do manifesto; vazio se nao der."""
    try:
        m = pd.read_parquet("artifacts/manifests/vindr_manifest.parquet")
    except Exception as e:                                   # sem pyarrow, sem manifesto
        print(f"  (sem manifesto: {e}); tamanho em mm aproximado com 0,085 mm/pixel")
        return pd.Series(dtype=float)
    idc = next((c for c in m.columns if c.lower() in ("image_id", "sopinstanceuid", "sop_instance_uid")), None)
    spc = [c for c in m.columns if "pixelspacing" in c.lower().replace("_", "")]
    if idc is None or not spc:
        print(f"  (manifesto sem colunas esperadas: {list(m.columns)[:12]}); usando 0,085 mm/pixel")
        return pd.Series(dtype=float)

    def first(v):
        try:
            if isinstance(v, str):
                v = v.strip("[]").replace("\\", ",").split(",")
            return float(list(v)[0])
        except Exception:
            return np.nan
    s = m[spc].apply(lambda r: next((first(v) for v in r if first(v) == first(v)), np.nan), axis=1)
    if idc != "image_id":
        print(f"  (id do manifesto: {idc})")
    return pd.Series(s.to_numpy(), index=m[idc].astype(str).str.replace(".dicom", "", regex=False))


def breast_frame(find: pd.DataFrame, preds: pd.DataFrame, term: str, spacing: pd.Series) -> pd.DataFrame:
    """Uma linha por mama do teste: pontuacao, se tem lesao do tipo, BI-RADS do achado,
    densidade, categoria da mama e o maior tamanho de caixa (mm)."""
    f = find.copy()
    f["is_kind"] = f.finding_categories.astype(str).str.contains(term, regex=False) & f.xmin.notna()
    sp = f.image_id.map(spacing).fillna(0.085)
    f["size_mm"] = np.where(f.is_kind, np.maximum(f.xmax - f.xmin, f.ymax - f.ymin) * sp, np.nan)
    f["fb"] = np.where(f.is_kind, f.finding_birads.astype(str).str.extract(r"(\d)")[0], None)
    score = preds.groupby("image_id").score.max()
    f["score"] = f.image_id.map(score).fillna(0.0)
    g = f.groupby(["study_id", "laterality"])
    br = g.agg(score=("score", "max"), lesao=("is_kind", "any"), size_mm=("size_mm", "max"),
               densidade=("breast_density", "first"), categoria=("breast_birads", "first"),
               tem_caixa=("xmin", lambda s: s.notna().any())).reset_index()
    br["birads_achado"] = g.fb.agg(lambda s: max([x for x in s if isinstance(x, str)], default=None)).to_numpy()
    br["densidade"] = br.densidade.astype(str).str.replace("DENSITY ", "", regex=False)
    br["categoria"] = br.categoria.astype(str).str.extract(r"(\d)")[0]
    return br


def rate_table(df: pd.DataFrame, by: str, hit: str = "detectada") -> pd.DataFrame:
    t = df.groupby(by, observed=True)[hit].agg(n="size", acertos="sum")
    t["taxa"] = (t.acertos / t.n).round(3)
    return t


def main() -> None:
    from src.config import dset, paths
    thr = yaml.safe_load(Path("configs/verifier_thresholds.yaml").read_text(encoding="utf-8"))
    find = pd.read_csv(dset("vindr") / paths()["vindr"]["finding_csv"])
    find = find[find.split == "test"]
    spacing = spacing_table()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, k in KINDS.items():
        pr = pd.read_csv(k["preds"])
        if "cls" in pr:
            pr = pr[pr.cls == 0]
        br = breast_frame(find, pr, k["term"], spacing)
        br["detectada"] = br.score >= thr[k["thr"]]
        pos = br[br.lesao].copy()
        pos["tamanho"] = pd.cut(pos.size_mm, k["bins"], labels=k["labels"], right=False)
        print(f"\n===== {name} | limiar {thr[k['thr']]:.4f} | {len(pos)} mamas com lesao, "
              f"{int(pos.detectada.sum())} detectadas ({pos.detectada.mean():.1%})")
        for col, title in (("birads_achado", "BI-RADS do achado"), ("densidade", "densidade"),
                           ("tamanho", "maior caixa")):
            print(f"\n-- deteccao por {title}\n" + rate_table(pos, col).to_string())
        neg = br[~br.tem_caixa & br.categoria.isin(["1", "2"])]
        print(f"\n-- alerta falso nas mamas de categoria 1 e 2 sem caixa ({len(neg)}), por densidade\n"
              + rate_table(neg, "densidade").to_string())
        print(f"   tamanho mediano da caixa: detectadas {pos[pos.detectada].size_mm.median():.1f} mm | "
              f"perdidas {pos[~pos.detectada].size_mm.median():.1f} mm")
        br.to_csv(OUT / f"atributos_{name}.csv", index=False)
    print(f"\n-> {OUT}/atributos_*.csv")


if __name__ == "__main__":
    main()
