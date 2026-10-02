import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "build_pairs", Path(__file__).resolve().parents[1] / "scripts" / "build_pairs.py")
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)


def _units():
    return pd.DataFrame({
        "unidade": [f"u{i}" for i in range(8)],
        "grupo": ["p0", "p0", "p1", "p2", "p3", "p4", "p5", "p5"],
        "lado": [1, 0, 0, 0, 1, 0, 1, 0],
        "categoria": ["4", "2", "1", "3", "5", "2", "4c", "1"],
    })


def test_one_original_per_unit():
    p = bp.make_pairs(_units(), k=2)
    o = p[p.tipo == "original"]
    assert sorted(o.unidade) == sorted(_units().unidade)
    assert (o.fonte == o.unidade).all() and (o.direcao == "consistente").all()


def test_swaps_cross_boundary_and_other_group():
    u = _units().set_index("unidade")
    p = bp.make_pairs(_units(), k=3)
    s = p[p.tipo == "trocado"]
    assert len(s) > 0
    for r in s.itertuples():
        assert u.loc[r.fonte, "lado"] != r.lado_imagem
        assert u.loc[r.fonte, "grupo"] != r.grupo
        assert r.categoria == u.loc[r.fonte, "categoria"]
        assert r.direcao == ("rebaixada" if r.lado_imagem == 1 else "elevada")


def test_k_distinct_donors_and_cap():
    p = bp.make_pairs(_units(), k=2)
    s = p[p.tipo == "trocado"]
    assert (s.groupby("unidade").fonte.nunique() == s.groupby("unidade").size()).all()
    assert s.groupby("unidade").size().max() <= 2
    # u0 (lado 1, grupo p0) tem 4 doadores possiveis de lado 0 fora de p0: k=10 usa os 4
    p10 = bp.make_pairs(_units(), k=10)
    assert (p10[(p10.tipo == "trocado")].unidade == "u0").sum() == 4


def test_deterministic_and_order_independent():
    a = bp.make_pairs(_units(), k=2)
    b = bp.make_pairs(_units().sample(frac=1, random_state=3), k=2)
    pd.testing.assert_frame_equal(a, b)
    c = bp.make_pairs(_units(), k=2, seed=1)
    assert not a.equals(c) or len(a[a.tipo == "trocado"]) == 0


def test_rejects_bad_input():
    u = _units()
    u.loc[0, "lado"] = 2
    try:
        bp.make_pairs(u, k=1)
    except ValueError:
        pass
    else:
        raise AssertionError("lado invalido deveria falhar")
