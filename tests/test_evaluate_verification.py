import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "evaluate_verification", Path(__file__).resolve().parents[1] / "scripts" / "evaluate_verification.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

THR = ev.Thresholds(mass=0.5, calc=0.5, cls_ge4=0.5)


def _cls(rows):
    return pd.DataFrame([{"image_id": i, "pb0": 0, "pb1": 0, "pb2": 1 - p, "pb3": p, "pb4": 0} for i, p in rows])


def test_unit_images_aggregation():
    img = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "unidade": ["E"] * 4,
                        "mama": ["E_L", "E_L", "E_R", "E_R"]})
    mass = pd.DataFrame({"image_id": ["a", "c", "c"], "score": [0.2, 0.7, 0.9], "cls": [0, 0, 1]})
    calc = pd.DataFrame({"image_id": ["b"], "score": [0.4]})
    cls = _cls([("a", 0.1), ("b", 0.3), ("c", 0.6), ("d", 0.8)])
    u = ev.unit_images(img, mass, calc, cls)["E"]
    assert u.mass_score == 0.7          # classe 1 (outra) ignorada
    assert u.calc_score == 0.4
    assert np.isclose(u.p_ge4, 0.7)     # media por mama (0,2 e 0,7), maior das mamas


def _pairs():
    # unidade X: imagem 4/5 (achado forte); unidade Y: imagem 1 a 3 (limpa)
    return pd.DataFrame({
        "par_id": range(4), "unidade": ["X", "Y", "X", "Y"], "grupo": ["gx", "gy", "gx", "gy"],
        "lado_imagem": [1, 0, 1, 0], "tipo": ["original", "original", "trocado", "trocado"],
        "direcao": ["consistente", "consistente", "rebaixada", "elevada"],
        "fonte": ["X", "Y", "Y", "X"], "categoria": ["5", "2", "2", "5"]})


def _images():
    return {"X": ev.BreastImage(0.9, 0.0, 0.9), "Y": ev.BreastImage(0.1, 0.1, 0.1)}


def test_alerts_and_metrics():
    al = ev.alerts_table(_pairs(), _images(), None, THR)
    assert al.verificavel.all()
    assert not al.loc[0, ["V1", "V2", "V4"]].any()          # X com 5: consistente
    assert not al.loc[1, ["V1", "V2", "V4"]].any()          # Y com 2: consistente
    assert al.loc[2, "V1"] and al.loc[2, "V4"]               # X com 2: rebaixada
    assert al.loc[3, "V2"] and al.loc[3, "V4"] and not al.loc[3, "V1"]   # Y com 5: elevada
    res = ev.evaluate(al, {"p": ["V1", "V3", "V4"], "d": ["V1"]}, n_boot=50)
    p = res["p"]
    assert p["especificidade"]["valor"] == 1.0
    assert p["encontra_rebaixada"]["valor"] == 1.0 and p["encontra_elevada"]["valor"] == 1.0
    assert res["d"]["encontra_elevada"]["valor"] == 0.0
    assert res["d"]["acuracia_balanceada"]["valor"] == 0.75
    lo, hi = p["especificidade"]["ic95"]
    assert lo <= 1.0 <= hi


def test_text_findings_by_source():
    f = {"X": [ev.ReportFinding("mass", True, "suspicious")], "Y": []}
    al = ev.alerts_table(_pairs(), _images(), f, THR)
    # par trocado 'elevada' usa o laudo X (achado suspeito) com categoria 5: sem V3
    assert not al.loc[3, "V3"]
    # se o laudo X tivesse categoria 2, V3 dispararia
    p = _pairs()
    p.loc[0, "categoria"] = "2"
    al2 = ev.alerts_table(p, _images(), f, THR)
    assert al2.loc[0, "V3"]


def test_bootstrap_by_group_counts():
    al = ev.alerts_table(pd.concat([_pairs()] * 3, ignore_index=True), _images(), None, THR)
    res = ev.evaluate(al, {"p": ["V1", "V4"]}, n_boot=20)
    assert res["p"]["n"] == {"orig": 6, "orig13": 3, "orig45": 3, "rebaixada": 3, "elevada": 3}


def test_paired_tests_runs():
    al = ev.alerts_table(_pairs(), _images(), {"X": [], "Y": []}, THR)
    t = ev.paired_tests(al, "principal (V1+V3+V4)", "so texto (V3)")
    assert t["trocados"]["b_only_a_correct"] == 2 and t["trocados"]["c_only_b_correct"] == 0


def test_threshold_per_group_and_lopo():
    pairs = pd.concat([_pairs().assign(grupo=lambda d: d.grupo + str(i), unidade=lambda d: d.unidade + str(i),
                                       fonte=lambda d: d.fonte + str(i)) for i in range(3)], ignore_index=True)
    imgs = {f"{u}{i}": im for i in range(3) for u, im in _images().items()}
    thr = ev.lopo_thresholds(pairs, imgs, alpha=0.05)
    assert set(thr) == set(pairs.grupo)
    # so as unidades Y (categoria 2, lado 0) entram: limiar = pontuacao delas
    assert all(abs(t.mass - 0.1) < 1e-9 and abs(t.cls_ge4 - 0.1) < 1e-9 for t in thr.values())
    al = ev.alerts_table(pairs, imgs, None, thr)
    assert len(al) == len(pairs)
