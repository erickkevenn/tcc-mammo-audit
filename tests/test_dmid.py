import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dmid import choose_group, image_key, parse_view, report_category

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_image_key_and_view():
    assert image_key("Img001.txt") == "IMG001" and image_key("IMG12.dcm") == "IMG012"
    assert image_key("lixo") is None
    assert parse_view("MLOLT ") == ("MLO", "L") and parse_view("CCRT") == ("CC", "R")
    assert parse_view("XX") == (None, None)


def test_report_category_rules():
    assert report_category("BIRADS: 3 and 5\n\nFindings:") == ("5", 2)
    assert report_category("BIRADS: 1\n") == ("1", 1)
    assert report_category("BIRADS: 4c). about two lesions (BIRADS 3)\n")[0] == "4c"
    assert report_category("BIRADS: 3). The inner quadrant lesion\n")[0] == "3"
    assert report_category("BIRADS: 2,3\n")[0] == "3"
    assert report_category("BIRADS: 4a, 4c\n")[0] == "4c"
    assert report_category("BIRADS: 0\n")[0] == "0"
    assert report_category("sem campo") == (None, 0)


def test_choose_group_rule():
    df = pd.DataFrame({"image_id": ["a", "b", "c"], "patient_id": ["p1", "p1", "p2"], "study_uid": ["s1", "s2", "s3"]})
    g, r = choose_group(df)
    assert r == "patient_id" and list(g) == ["p1", "p1", "p2"]
    df["patient_id"] = ["anon", "anon", "anon"]          # um valor so: cai para o estudo
    g, r = choose_group(df)
    assert r == "study_uid"
    df["study_uid"] = ["", "s", "s"]
    g, r = choose_group(df)
    assert r == "image_id" and list(g) == ["a", "b", "c"]


def test_split_groups_stratified_and_stable():
    dp = _load("dmid_pairs")
    units = pd.DataFrame({"unidade": [f"u{i}" for i in range(40)], "grupo": [f"g{i // 2}" for i in range(40)],
                          "lado": [1 if i < 12 else 0 for i in range(40)], "categoria": "2"})
    a = dp.split_groups(units, 0.5, 20260819)
    b = dp.split_groups(units.sample(frac=1, random_state=1), 0.5, 20260819)
    assert a == b and len(a) == 10
    side = units.groupby("grupo").lado.max()
    assert sum(side[list(a)] == 1) == 3                   # metade dos 6 grupos com 4/5


def test_calibrate_rule():
    ed = _load("evaluate_dmid")
    from src.verify.rules import BreastImage
    imgs = {f"u{i}": BreastImage(i / 100, i / 200, i / 100) for i in range(100)}
    units = pd.DataFrame({"unidade": [f"u{i}" for i in range(100)],
                          "categoria": ["1"] * 50 + ["3"] * 30 + ["4"] * 20,
                          "lado": [0] * 80 + [1] * 20})
    t = ed.calibrate(units, imgs, 0.05)
    assert np.isclose(t.mass, np.quantile(np.arange(50) / 100, 0.975))
    assert np.isclose(t.cls_ge4, np.quantile(np.arange(80) / 100, 0.95))


def test_to_gray_rgb():
    from src.preprocess.dicom_io import to_gray
    a = np.zeros((2, 2, 3), np.uint8)
    a[0, 0] = [100, 100, 100]
    a[1, 1] = [255, 0, 0]
    g = to_gray(a, "RGB")
    assert g.shape == (2, 2) and g.dtype == np.int32
    assert g[0, 0] == 100 and g[1, 1] == round(0.299 * 255) and g[0, 1] == 0
