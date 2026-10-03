import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ea = _load("error_analysis")
ci = _load("cls_test_ci")
cb = _load("eval_cbis_mass")


def _df(rows, cols):
    return pd.DataFrame(rows, columns=cols)


def test_match_errors_fp_fn():
    gts = _df([("a", 10, 10, 50, 50), ("b", 0, 0, 20, 20)], ["image_id", "x1", "y1", "x2", "y2"])
    preds = _df([("a", 20, 20, 40, 40, 0.9),      # acerta a caixa de a
                 ("a", 100, 100, 120, 120, 0.8),  # FP em a
                 ("a", 110, 110, 130, 130, 0.95), # FP em a, maior: e o que fica
                 ("b", 5, 5, 15, 15, 0.1),        # dentro de b, mas abaixo do limiar
                 ("c", 0, 0, 10, 10, 0.7)],       # imagem sem caixa: FP
                ["image_id", "x1", "y1", "x2", "y2", "score"])
    fp, fn = ea.match_errors(preds, gts, thr=0.5)
    assert list(fp.image_id) == ["a", "c"] and fp.score.iloc[0] == 0.95
    assert list(fn.image_id) == ["b"] and fn.melhor_score.iloc[0] == 0.1


def test_pick_and_tiles(tmp_path):
    fp = _df([("x", 0, 0, 1, 1, s) for s in np.linspace(1, 0, 40)], ["image_id", "x1", "y1", "x2", "y2", "score"])
    fn = _df([("y", 0, 0, 1, 1, 0.0)] * 5, ["image_id", "x1", "y1", "x2", "y2", "melhor_score"])
    fps, fns = ea.pick(fp, fn, n=30)
    assert len(fps) == 30 and fps.score.iloc[0] == 1 and len(fns) == 5
    img = np.zeros((300, 200), np.uint8)
    t = ea.crop_tile(img, (50, 50, 80, 90), [], ea.RED, min_side=96, label="1")
    assert t.shape == (200, 200, 3)
    ea.sheet([t, t, t], tmp_path / "s.png", cols=2)
    assert (tmp_path / "s.png").exists()


def test_cls_boot_shapes():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(30):
        for lat in "LR":
            b = int(rng.integers(0, 5))
            p = np.full(5, 0.05); p[b] = 0.8
            rows.append({"study_id": f"s{s}", "laterality": lat, "birads": b, "density": b % 3,
                         **{f"pb{i}": p[i] for i in range(5)}, "pd0": 0.2, "pd1": 0.6, "pd2": 0.2})
    br = pd.DataFrame(rows)
    t = ci.boot_cls(br, n_boot=20)
    assert "birads_qwk" in t.index and "mamas" not in t.index
    assert (t.ic_lo <= t.valor + 1e-9).all() and (t.valor <= t.ic_hi + 1e-9).all()


def test_cbis_names(tmp_path):
    d = tmp_path / "img"
    d.mkdir()
    for n in ("Mass-Test_P_00016_LEFT_CC", "Mass-Test_P_00016_RIGHT_MLO", "lixo"):
        (d / f"{n}.png").write_bytes(b"")
    t = cb.image_table(d)
    assert list(t.study_id) == ["P_00016", "P_00016"] and list(t.laterality) == ["L", "R"]
