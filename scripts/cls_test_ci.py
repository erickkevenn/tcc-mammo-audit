"""IC 95% das metricas do classificador por mama, por bootstrap de ESTUDOS (S15).

Le as predicoes ja salvas (artifacts/cls_runs/<run>/preds_<split>.csv): nao roda
o modelo de novo. Salva metrics_<split>_ic.json ao lado.

Uso:  python scripts/cls_test_ci.py               (teste)
      python scripts/cls_test_ci.py --split val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.cls_metrics import breast_aggregate, cls_metrics       # noqa: E402


def boot_cls(br: pd.DataFrame, n_boot: int = 1000, seed: int = 20260819) -> pd.DataFrame:
    point = cls_metrics(br)
    groups = [g.index.to_numpy() for _, g in br.groupby("study_id")]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        rows.append(cls_metrics(br.loc[np.concatenate([groups[i] for i in pick])]))
    b = pd.DataFrame(rows)
    out = pd.DataFrame({"valor": pd.Series(point), "ic_lo": b.quantile(0.025), "ic_hi": b.quantile(0.975)})
    return out.drop(index="mamas")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cls_b2_imagenet")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    d = Path("artifacts/cls_runs") / args.run
    br = breast_aggregate(pd.read_csv(d / f"preds_{args.split}.csv")).reset_index(drop=True)
    tab = boot_cls(br, args.n_boot)
    print(f"{args.run} | {args.split} | {len(br)} mamas, {br.study_id.nunique()} estudos | "
          f"{args.n_boot} reamostragens de estudos")
    print(tab.round(3).to_string())
    out = d / f"metrics_{args.split}_ic.json"
    out.write_text(json.dumps(tab.round(4).to_dict(orient="index"), indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
