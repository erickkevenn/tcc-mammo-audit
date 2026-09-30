"""Splits por PACIENTE. Split por imagem garante vazamento em mamografia:
cada exame tem 4 vistas da mesma paciente e a mesma lesao aparece em CC e MLO.

Regra do projeto: use o split OFICIAL do VinDr (4.000/1.000 exames) para que
seus numeros sejam comparaveis a literatura. Use esta funcao apenas quando
criar um split proprio (ex.: fundir massa+calc do CBIS-DDSM, ou dividir os
laudos do INbreast em calibracao/teste da auditoria).
"""
from __future__ import annotations
import hashlib
import pandas as pd


def _bucket(key: str, salt: str, n: int = 1000) -> int:
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:8], 16) % n


def patient_split(
    df: pd.DataFrame,
    patient_col: str,
    fractions: dict[str, float] | None = None,
    salt: str = "tcc-mammo-2026",
) -> pd.DataFrame:
    """Atribui split deterministicamente por paciente (mesmo paciente -> mesmo split).

    Deterministico por hash: reproduzivel entre maquinas sem salvar estado.
    """
    fractions = fractions or {"train": 0.7, "val": 0.15, "test": 0.15}
    total = sum(fractions.values())
    assert abs(total - 1.0) < 1e-6, f"fracoes somam {total}"

    edges, acc = [], 0.0
    for name, frac in fractions.items():
        acc += frac
        edges.append((name, acc * 1000))

    def assign(pid: str) -> str:
        b = _bucket(str(pid), salt)
        for name, edge in edges:
            if b < edge:
                return name
        return edges[-1][0]

    out = df.copy()
    out["split"] = out[patient_col].map(assign)
    return out


def assert_no_leakage(df: pd.DataFrame, patient_col: str, split_col: str = "split") -> None:
    """Falha alto se qualquer paciente aparecer em mais de um split."""
    counts = df.groupby(patient_col)[split_col].nunique()
    bad = counts[counts > 1]
    if len(bad):
        raise AssertionError(
            f"VAZAMENTO: {len(bad)} paciente(s) em mais de um split. Ex.: {list(bad.index[:5])}"
        )


def stratification_report(df: pd.DataFrame, split_col: str, strata: list[str]) -> pd.DataFrame:
    """Proporcao de cada estrato por split -- cole isto na Tabela T2 do TCC."""
    frames = []
    for s in strata:
        t = (
            df.groupby([split_col, s]).size().rename("n").reset_index()
        )
        t["frac"] = t.groupby(split_col)["n"].transform(lambda x: x / x.sum())
        t["stratum"] = s
        t = t.rename(columns={s: "value"})
        frames.append(t[[split_col, "stratum", "value", "n", "frac"]])
    return pd.concat(frames, ignore_index=True)
