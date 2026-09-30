"""Z0 -- 'sempre sem erro'. Mostra o PRIOR.

Sem este baseline, nenhum numero de acuracia tem interpretacao: se 90% dos
laudos estao corretos, um sistema burro acerta 90%. Reporte Z0 na mesma tabela.
"""
from __future__ import annotations


def predict(cases: list[dict]) -> list[list]:
    return [[] for _ in cases]


def accuracy(cases: list[dict]) -> float:
    clean = sum(1 for c in cases if c.get("expected_alert") is None)
    return clean / max(len(cases), 1)
