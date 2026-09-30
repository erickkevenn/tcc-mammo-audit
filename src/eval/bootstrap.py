"""Intervalos de confianca por bootstrap.

REGRA CRITICA: reamostre EXAMES (ou pacientes) com reposicao -- nunca lesoes
nem imagens isoladas. As vistas CC e MLO da mesma mama sao correlacionadas, e
reamostrar imagens SUBESTIMA a variancia.

Realismo brutal, planeje para isso: para um IC de 1% de largura em metrica
continua por caso sao necessarios ~100-200 casos; para ACURACIA (metrica
discreta), ~10.000. Com 50-300 exames voce NAO tera IC estreito. Reporte a
largura do IC como resultado, nao como incomodo, e escreva 'nao foi possivel
descartar diferenca' -- nunca 'os modelos sao equivalentes'.
"""
from __future__ import annotations
from typing import Callable, Sequence
import numpy as np


def bootstrap_ci(units: Sequence, metric_fn: Callable[[Sequence], float],
                 n_boot: int = 2000, alpha: float = 0.05,
                 method: str = "bca", seed: int = 20260819) -> dict:
    """units: lista de EXAMES (cada elemento agrega todas as imagens do exame)."""
    rng = np.random.default_rng(seed)
    units = list(units)
    n = len(units)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    point = float(metric_fn(units))
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric_fn([units[i] for i in idx])

    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    if method == "percentile":
        lo, hi = np.percentile(stats, [lo_p, hi_p])
    else:                                     # BCa: melhor para AP e F1 em classe rara
        from scipy.stats import norm as _norm
        prop = float(np.mean(stats < point))
        prop = min(max(prop, 1e-6), 1 - 1e-6)
        z0 = _norm.ppf(prop)
        jack = np.array([metric_fn(units[:i] + units[i + 1:]) for i in range(n)])
        jbar = jack.mean()
        num = float(((jbar - jack) ** 3).sum())
        den = 6.0 * (float(((jbar - jack) ** 2).sum()) ** 1.5)
        acc = num / den if den != 0 else 0.0
        z_lo, z_hi = _norm.ppf(alpha / 2), _norm.ppf(1 - alpha / 2)

        def adj(z):
            return _norm.cdf(z0 + (z0 + z) / max(1 - acc * (z0 + z), 1e-9))

        lo, hi = np.percentile(stats, [100 * adj(z_lo), 100 * adj(z_hi)])

    return {"point": point, "lo": float(lo), "hi": float(hi),
            "width": float(hi - lo), "n": n, "n_boot": n_boot, "method": method}


def fmt(ci: dict, digits: int = 3) -> str:
    """Formato para colar direto na tabela do TCC."""
    return f"{ci['point']:.{digits}f} (IC 95% {ci['lo']:.{digits}f}-{ci['hi']:.{digits}f}; n={ci['n']})"
