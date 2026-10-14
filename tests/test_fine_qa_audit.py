import numpy as np
import pytest

from scripts.audit_fine_grouped_qa import metrics, scene_bootstrap


def test_metrics_and_changed_condition_pair_counts_are_independent():
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], np.float64)
    y = np.array([0, 1, 2, 2], np.int64)
    m = metrics(p, np.log(p), y)
    assert m["accuracy"] == 0.75 and m["both_correct"] == 0.5
    assert m["changed"]["both_correct"] == 1 and m["unchanged"]["both_correct"] == 0
    assert m["initial"]["accuracy"] == 0.5 and m["final"]["accuracy"] == 1
    assert abs(m["ece"] - 0.05) < 1e-12
    assert abs(m["nll"] - (-3 * np.log(0.8) - np.log(0.1)) / 4) < 1e-12


def test_scene_bootstrap_keeps_two_conditions_together():
    a = np.tile([0.8, 0.1, 0.1], (4, 1))
    b = np.tile([0.1, 0.8, 0.1], (4, 1))
    y = np.array([0, 0, 1, 1], np.int64)
    m = scene_bootstrap(a, np.log(a), b, np.log(b), y)
    assert m["accuracy"]["difference"] == 0
    assert m["accuracy"]["interval95"] == [-1, 1]
    assert m["videos"] == 2 and m["repetitions"] == 10000 and m["seed"] == 17
    with pytest.raises(ValueError):
        metrics(a, np.log(a), y[:3])
