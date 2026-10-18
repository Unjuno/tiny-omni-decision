import numpy as np
import pytest

from scripts.summarize_condition_binding import grouped_interval, score_difference


def test_bootstrap_keeps_correlated_questions_in_whole_state_groups():
    values = np.array([[1.0, -2.0]] * 4 + [[-1.0, 2.0]] * 4)
    groups = ["a"] * 4 + ["b"] * 4
    result = grouped_interval(values, groups)
    assert result["difference"] == [0.0, 0.0]
    assert result["lower"] == [-1.0, -2.0]
    assert result["upper"] == [1.0, 2.0]
    assert result == grouped_interval(values[::-1], groups[::-1])
    assert result["groups"] == 2 and result["questions"] == 8


def test_same_selected_model_has_exact_zero_paired_difference():
    rows = [dict(options=["a", "b"], target="a", group=g) for g in ("a", "b")]
    scores = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = score_difference(rows, scores, scores.copy())
    assert result["difference"] == result["lower"] == result["upper"] == [0.0, 0.0]


def test_bootstrap_rejects_nonfinite_or_unaligned_values():
    with pytest.raises(ValueError):
        grouped_interval([[float("nan")], [0.0]], ["a", "b"])
    with pytest.raises(ValueError):
        grouped_interval([[0.0], [1.0]], ["a"])
    with pytest.raises(ValueError):
        grouped_interval([[0.0], [1.0]], ["a", "a"])
