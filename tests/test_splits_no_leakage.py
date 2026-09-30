"""Split por paciente nunca deve vazar. Este teste roda em CI e antes de treinar."""
import pandas as pd
import pytest

from src.data.splits import assert_no_leakage, patient_split, stratification_report


def _df(n=400):
    return pd.DataFrame({
        "patient_id": [f"P{i//4:04d}" for i in range(n)],   # 4 vistas por paciente
        "view": ["CC", "MLO"] * (n // 2),
        "birads": [1, 2, 3, 4, 5] * (n // 5),
    })


def test_patient_split_is_deterministic():
    a = patient_split(_df(), "patient_id")
    b = patient_split(_df(), "patient_id")
    assert (a["split"].values == b["split"].values).all()


def test_no_patient_in_two_splits():
    out = patient_split(_df(), "patient_id")
    assert_no_leakage(out, "patient_id")


def test_leakage_is_detected():
    out = patient_split(_df(), "patient_id")
    out.loc[0, "split"] = "test"
    out.loc[1, "split"] = "train"
    with pytest.raises(AssertionError):
        assert_no_leakage(out, "patient_id")


def test_fractions_roughly_respected():
    out = patient_split(_df(2000), "patient_id",
                        {"train": 0.7, "val": 0.15, "test": 0.15})
    frac = out["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.7) < 0.06


def test_stratification_report_runs():
    out = patient_split(_df(), "patient_id")
    rep = stratification_report(out, "split", ["birads"])
    assert {"split", "stratum", "value", "n", "frac"} <= set(rep.columns)
