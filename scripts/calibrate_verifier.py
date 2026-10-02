"""Calibra os limiares das regras V1, V2 e V4 na VALIDACAO do VinDr-Mammo (S12).

No VinDr, a categoria BI-RADS de cada mama foi dada pelos radiologistas da
propria base: cada mama com a sua categoria e um par CONSISTENTE por definicao
(par original). Os limiares sao escolhidos para que, nesses pares, cada fonte de
alerta fique em torno de --alpha (padrao 5%):
  * t_mass e t_calc: percentil (1 - alpha/2) das pontuacoes das mamas de
    categoria 1 e 2 (a V1 so olha essas mamas; metade do alpha para cada detector);
  * t_cls: percentil (1 - alpha) de P(4)+P(5) nas mamas de categoria 1 a 3.
Os limiares vao para configs/verifier_thresholds.yaml e ficam CONGELADOS antes do teste.

Depois, ainda na validacao, mede o comportamento com PARES TROCADOS: cada mama
recebe a categoria de outra mama do lado oposto da fronteira de biopsia (sem
escrever nem editar laudo). Isso e desenvolvimento; o numero final sai do teste
oficial, uma vez so, na S14.

Uso:  python scripts/calibrate_verifier.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import dset, paths                                  # noqa: E402
from src.data.vindr_breast import cls_table                         # noqa: E402
from src.eval.cls_metrics import breast_aggregate                   # noqa: E402
from src.verify.rules import BreastImage, Thresholds, biopsy_side, verify_breast  # noqa: E402

MASS_PREDS = "artifacts/eval/det_mass_B0_v2_neg1_val_preds.csv"
CALC_PREDS = "artifacts/eval/det_calc_B0_val_preds.csv"
CLS_PREDS = "artifacts/cls_runs/cls_b2_imagenet/preds_val.csv"


def breast_table() -> pd.DataFrame:
    """Uma linha por mama da validacao: categoria e as tres pontuacoes da imagem."""
    cfg = paths()["vindr"]
    tab = cls_table(pd.read_csv(dset("vindr") / cfg["breast_csv"]))
    tab = tab[tab.split == "val"].copy()
    tab["category"] = (tab.birads + 1).astype(str)

    def per_image(path, cls=None):
        p = pd.read_csv(path)
        if cls is not None and "cls" in p:
            p = p[p.cls == cls]
        return p.groupby("image_id").score.max()

    m, c = per_image(MASS_PREDS, 0), per_image(CALC_PREDS)
    tab["mass_seen"] = tab.image_id.isin(pd.read_csv(MASS_PREDS).image_id.unique())
    tab["mass"] = tab.image_id.map(m).fillna(0.0)
    tab["calc"] = tab.image_id.map(c).fillna(0.0)
    br = (tab.groupby(["study_id", "laterality"])
             .agg(category=("category", "first"), mass=("mass", "max"), calc=("calc", "max"),
                  mass_views=("mass_seen", "sum"))
             .reset_index())
    cls = breast_aggregate(pd.read_csv(CLS_PREDS))
    cls["p_ge4"] = cls.pb3 + cls.pb4
    br = br.merge(cls[["study_id", "laterality", "p_ge4"]], on=["study_id", "laterality"], how="left")
    br["side"] = br.category.map(biopsy_side)
    return br


def alerts_for(br: pd.DataFrame, categories: pd.Series, thr: Thresholds) -> pd.DataFrame:
    rows = []
    for r, cat in zip(br.itertuples(index=False), categories):
        a = verify_breast(cat, BreastImage(r.mass, r.calc, r.p_ge4), None, thr)
        codes = {x.rule for x in a}
        rows.append({f"V{k}": f"V{k}" in codes for k in (1, 2, 4)})
    out = pd.DataFrame(rows, index=br.index)
    out["any"] = out.any(axis=1)
    return out


def swap_categories(br: pd.DataFrame, seed: int = 20260819) -> pd.Series:
    """Categoria de outra mama do lado oposto da fronteira, sorteada com semente fixa."""
    rng = np.random.default_rng(seed)
    low = br.loc[br.side == 0, "category"].to_numpy()
    high = br.loc[br.side == 1, "category"].to_numpy()
    return pd.Series([rng.choice(high) if s == 0 else rng.choice(low) for s in br.side], index=br.index)


def boot_ci(br, orig, swap, n=1000, seed=20260819):
    rng = np.random.default_rng(seed)
    groups = {s: np.flatnonzero(br.study_id.to_numpy() == s) for s in br.study_id.unique()}
    keys = list(groups)
    stats = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        stats.append((1 - orig["any"].to_numpy()[idx].mean(), swap["any"].to_numpy()[idx].mean()))
    s = np.array(stats)
    return np.percentile(s, [2.5, 97.5], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    br = breast_table()
    print(f"mamas na validacao: {len(br)} | sem nenhuma vista no detector de massas: "
          f"{int((br.mass_views == 0).sum())}")
    print("categorias:", br.category.value_counts().sort_index().to_dict())

    low12 = br[br.category.isin(["1", "2"])]
    low13 = br[br.side == 0]
    thr = Thresholds(mass=float(np.quantile(low12.mass, 1 - args.alpha / 2)),
                     calc=float(np.quantile(low12.calc, 1 - args.alpha / 2)),
                     cls_ge4=float(np.quantile(low13.p_ge4.dropna(), 1 - args.alpha)))
    print(f"\nlimiares (alpha {args.alpha:.0%}): massa {thr.mass:.4f} | calcificacao {thr.calc:.4f}"
          f" | P(4)+P(5) {thr.cls_ge4:.4f}")

    orig = alerts_for(br, br.category, thr)
    swapped_cat = swap_categories(br)
    swap = alerts_for(br, swapped_cat, thr)
    hi, lo = br.side == 1, br.side == 0

    def pct(x):
        return f"{x.mean():.1%} ({int(x.sum())}/{len(x)})"

    print("\nPARES ORIGINAIS (consistentes): alertas falsos")
    for g, mask in (("categoria 1 a 3", lo), ("categoria 4 e 5", hi), ("todas", lo | hi)):
        o = orig[mask]
        print(f"  {g:16s} V1 {pct(o.V1)} | V2 {pct(o.V2)} | V4 {pct(o.V4)} | qualquer {pct(o['any'])}")
    print("\nPARES TROCADOS (inconsistentes): inconsistencias encontradas")
    for g, mask, desc in (("rebaixada", hi, "imagem de mama 4/5 com categoria 1 a 3"),
                          ("elevada", lo, "imagem de mama 1 a 3 com categoria 4/5")):
        s = swap[mask]
        print(f"  {g:10s} ({desc}): V1 {pct(s.V1)} | V2 {pct(s.V2)} | V4 {pct(s.V4)} | qualquer {pct(s['any'])}")
    ci = boot_ci(br, orig, swap)
    spec, sens = 1 - orig["any"].mean(), swap["any"].mean()
    print(f"\nespecificidade nos originais: {spec:.3f} [{ci[0, 0]:.3f} a {ci[1, 0]:.3f}]")
    print(f"sensibilidade nos trocados:   {sens:.3f} [{ci[0, 1]:.3f} a {ci[1, 1]:.3f}]")
    print("  (a maioria dos trocados e 'elevada', porque a validacao tem muito mais mamas 1 a 3)")

    cfg = {"calibrado_em": "validacao VinDr-Mammo (mesma divisao dos modelos)",
           "alpha": args.alpha,
           "regra": "percentil (1 - alpha/2) nas mamas 1-2 para cada detector; (1 - alpha) de P(4)+P(5) nas mamas 1-3",
           "mass": round(thr.mass, 6), "calc": round(thr.calc, 6), "cls_ge4": round(thr.cls_ge4, 6),
           "modelos": {"massa": "det_mass_B0_v2_neg1", "calcificacao": "det_calc_B0",
                       "classificador": "cls_b2_imagenet"}}
    Path("configs/verifier_thresholds.yaml").write_text(
        "# Limiares CONGELADOS das regras V1, V2 e V4 (scripts/calibrate_verifier.py).\n"
        "# Nao recalibrar depois de olhar o teste.\n" + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    out = Path("artifacts/verify")
    out.mkdir(parents=True, exist_ok=True)
    res = {"thresholds": cfg, "n_mamas": len(br), "especificidade": [spec, *ci[:, 0]],
           "sensibilidade_trocados": [sens, *ci[:, 1]],
           "originais": {k: float(orig.loc[lo | hi, k].mean()) for k in orig},
           "trocados_rebaixada": {k: float(swap.loc[hi, k].mean()) for k in swap},
           "trocados_elevada": {k: float(swap.loc[lo, k].mean()) for k in swap}}
    (out / "calibracao_val.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nlimiares -> configs/verifier_thresholds.yaml | resultado -> {out / 'calibracao_val.json'}")


if __name__ == "__main__":
    main()
