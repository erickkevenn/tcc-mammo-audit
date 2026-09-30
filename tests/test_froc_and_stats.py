"""FROC e estatistica: casos degenerados nao devem explodir."""
import numpy as np

from src.eval.bootstrap import bootstrap_ci
from src.eval.froc import center_in_box, froc_curve, iou
from src.eval.stats import (cohen_kappa, delong_test, interpret_kappa, mcnemar,
                            quadratic_weighted_kappa)


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_center_criterion_for_tiny_lesions():
    assert center_in_box((9, 9, 11, 11), (0, 0, 20, 20)) is True
    assert center_in_box((100, 100, 102, 102), (0, 0, 20, 20)) is False


def test_perfect_detector_reaches_full_sensitivity():
    preds = {"a": [((0, 0, 10, 10), 0.9)], "b": [((5, 5, 15, 15), 0.8)]}
    gts = {"a": [(0, 0, 10, 10)], "b": [(5, 5, 15, 15)]}
    out = froc_curve(preds, gts)
    assert out["sensitivity"][-1] == 1.0
    assert out["operating_points"]["sens@0.5fppi"] == 1.0


def test_froc_handles_empty_predictions():
    out = froc_curve({"a": []}, {"a": [(0, 0, 10, 10)]})
    assert out["n_lesions"] == 1 and out["sensitivity"] == []


def test_bootstrap_ci_brackets_the_point_estimate():
    units = [{"hit": 1} for _ in range(70)] + [{"hit": 0} for _ in range(30)]
    ci = bootstrap_ci(units, lambda u: float(np.mean([x["hit"] for x in u])),
                      n_boot=300, method="percentile")
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert 0.55 < ci["point"] < 0.85


def test_mcnemar_exact_for_small_counts():
    a = [1] * 20 + [0] * 5
    b = [1] * 18 + [0] * 7
    out = mcnemar(a, b)
    assert out["test"] == "exact" and 0.0 <= out["p"] <= 1.0


def test_delong_runs_and_orders_aucs():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(50), np.zeros(50)].astype(int)
    good = np.r_[rng.normal(1.5, 1, 50), rng.normal(0, 1, 50)]
    weak = np.r_[rng.normal(0.3, 1, 50), rng.normal(0, 1, 50)]
    out = delong_test(y, good, weak)
    assert out["auc_a"] > out["auc_b"] and 0 <= out["p"] <= 1


def test_quadratic_kappa_penalises_distant_errors_more():
    truth = [0, 1, 2, 3, 4]
    near = [0, 1, 2, 3, 3]
    far = [0, 1, 2, 3, 0]
    assert quadratic_weighted_kappa(truth, near) > quadratic_weighted_kappa(truth, far)


def test_cohen_kappa_reports_raw_agreement_and_marginals():
    out = cohen_kappa([1, 1, 0, 0, 1], [1, 0, 0, 0, 1])
    assert "raw_agreement" in out and "marginals_a" in out
    assert interpret_kappa(0.85) == "quase perfeita"
