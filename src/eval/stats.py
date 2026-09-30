"""Testes estatisticos: DeLong, McNemar exato e kappas.

Quando usar cada um:
  DeLong  -> comparar duas AUROC CORRELACIONADAS (mesmos casos).
  McNemar -> comparar acuracia pareada em um PONTO DE OPERACAO fixo.
             Use a versao EXATA (binomial) se b+c < 25.
  kappa quadratico -> BI-RADS (e ORDINAL; kappa simples desperdica informacao).
  Fleiss  -> 3+ avaliadores.
Interprete kappa pelos limiares de Landis & Koch (1977) CITANDO a fonte, e
reporte tambem concordancia bruta e prevalencias marginais: kappa colapsa com
prevalencia desbalanceada ('paradoxo do kappa').
"""
from __future__ import annotations
import numpy as np

LANDIS_KOCH = [(0.20, "leve"), (0.40, "razoavel"), (0.60, "moderada"),
               (0.80, "substancial"), (1.01, "quase perfeita")]


def interpret_kappa(k: float) -> str:
    if k <= 0:
        return "pobre"
    for edge, label in LANDIS_KOCH:
        if k <= edge:
            return label
    return "quase perfeita"


# ------------------------------------------------------------------- DeLong
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    tr = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        tr[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = tr
    return out


def delong_test(y_true, score_a, score_b) -> dict:
    """p-valor bicaudal para H0: AUC_a == AUC_b (DeLong et al., Biometrics 1988)."""
    from scipy.stats import norm

    y = np.asarray(y_true).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        raise ValueError("precisa de casos positivos e negativos")

    preds = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    k = preds.shape[0]
    aucs = np.empty(k)
    v01 = np.empty((k, m))
    v10 = np.empty((k, n))
    for r in range(k):
        p, q = preds[r, pos], preds[r, neg]
        tz = _midrank(np.r_[p, q])
        tx, ty = _midrank(p), _midrank(q)
        aucs[r] = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        v01[r] = (tz[:m] - tx) / n
        v10[r] = 1.0 - (tz[m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    s = np.atleast_2d(s)
    contrast = np.array([[1.0, -1.0]])
    var = float(np.squeeze(contrast @ s @ contrast.T))
    if var <= 0:
        return {"auc_a": aucs[0], "auc_b": aucs[1], "z": 0.0, "p": 1.0}
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": z,
            "p": float(2 * (1 - norm.cdf(abs(z))))}


# ------------------------------------------------------------------ McNemar
def mcnemar(correct_a, correct_b, exact: bool | None = None) -> dict:
    a = np.asarray(correct_a).astype(bool)
    b = np.asarray(correct_b).astype(bool)
    b01 = int(np.sum(a & ~b))          # A acerta, B erra
    c10 = int(np.sum(~a & b))
    n = len(a)
    total = b01 + c10
    use_exact = (total < 25) if exact is None else exact
    if total == 0:
        p = 1.0
    elif use_exact:
        from scipy.stats import binomtest
        p = float(binomtest(b01, total, 0.5).pvalue)
    else:
        from scipy.stats import chi2
        stat = (abs(b01 - c10) - 1) ** 2 / total
        p = float(1 - chi2.cdf(stat, 1))
    diff = (a.mean() - b.mean())
    se = np.sqrt(max(total, 1)) / n
    return {"b_only_a_correct": b01, "c_only_b_correct": c10, "n": n,
            "acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "diff": float(diff), "ci95": (float(diff - 1.96 * se), float(diff + 1.96 * se)),
            "p": p, "test": "exact" if use_exact else "chi2"}


# -------------------------------------------------------------------- kappas
def quadratic_weighted_kappa(a, b, n_classes: int | None = None) -> float:
    """Para BI-RADS. Penaliza erro de 2 niveis 4x mais que erro de 1 nivel."""
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    k = n_classes or int(max(a.max(), b.max()) + 1)
    obs = np.zeros((k, k))
    for i, j in zip(a, b):
        obs[i, j] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float)
    w /= max((k - 1) ** 2, 1)
    ha, hb = np.bincount(a, minlength=k), np.bincount(b, minlength=k)
    exp = np.outer(ha, hb).astype(float)
    exp *= obs.sum() / exp.sum()
    return float(1 - (w * obs).sum() / max((w * exp).sum(), 1e-9))


def cohen_kappa(a, b) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    n = obs.sum()
    po = np.trace(obs) / n
    pe = float((obs.sum(0) * obs.sum(1)).sum()) / (n * n)
    kappa = (po - pe) / max(1 - pe, 1e-9)
    return {"kappa": float(kappa), "raw_agreement": float(po),
            "expected": float(pe), "interpretation": interpret_kappa(kappa),
            "marginals_a": (obs.sum(1) / n).tolist(),
            "marginals_b": (obs.sum(0) / n).tolist()}


def fleiss_kappa(table: np.ndarray) -> float:
    """table[i, j] = numero de avaliadores que deram a categoria j ao item i."""
    table = np.asarray(table, float)
    n_items, _ = table.shape
    n_raters = table.sum(1)[0]
    p_j = table.sum(0) / (n_items * n_raters)
    p_i = (np.square(table).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    pbar, pebar = p_i.mean(), float(np.square(p_j).sum())
    return float((pbar - pebar) / max(1 - pebar, 1e-9))
