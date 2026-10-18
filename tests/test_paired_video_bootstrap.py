import numpy as np
import pytest

from scripts.paired_video_bootstrap import paired_video_bootstrap


def test_identical_predictions_have_zero_intervals():
    p = np.array([[0.8, 0.2], [0.3, 0.7]] * 3)
    result = paired_video_bootstrap(p, p, np.array([0, 1] * 3))
    assert result["accuracy"] == {"difference": 0.0, "interval": [0.0, 0.0]}
    assert result["nll"] == {"difference": 0.0, "interval": [0.0, 0.0]}
    assert result["replicates"] == 10000


def test_questions_remain_paired_and_sign_is_normal_minus_mismatch():
    normal = np.array([[0.9, 0.1], [0.9, 0.1]] * 4)
    mismatch = normal[:, ::-1].copy()
    # Each video has one gain and one loss: resampling individual questions
    # would incorrectly produce a nonzero interval.
    result = paired_video_bootstrap(normal, mismatch, np.array([0, 1] * 4))
    assert result["accuracy"]["interval"] == [0.0, 0.0]
    assert np.allclose(result["nll"]["interval"], 0)
    win = paired_video_bootstrap(normal, mismatch, np.zeros(8, dtype=int))
    assert win["accuracy"]["difference"] == 1
    assert win["nll"]["difference"] < 0
    assert win == paired_video_bootstrap(normal, mismatch, np.zeros(8, dtype=int))


@pytest.mark.parametrize("p,y", [
    ([[0.5, 0.5]], [0]),
    ([[float("nan"), 0.5]] * 2, [0, 0]),
    ([[0.8, 0.8]] * 2, [0, 0]),
    ([[0.5, 0.5]] * 2, [2, 0]),
])
def test_invalid_inputs_fail_closed(p, y):
    with pytest.raises(ValueError):
        paired_video_bootstrap(np.array(p), np.array(p), np.array(y))


def test_explicit_floor_matches_aggregate_nll_even_for_extreme_predictions():
    p = np.array([[1e-30, 1.0], [0.2, 0.8]])
    q = np.array([[0.5, 0.5], [1.0, 0.0]])
    y = np.array([0, 1])
    result = paired_video_bootstrap(p, q, y, probability_floor=1e-12)
    rows = np.arange(2)
    expected = (-np.log(np.clip(p[rows, y], 1e-12, 1)).mean()
                + np.log(np.clip(q[rows, y], 1e-12, 1)).mean())
    assert result['nll']['difference'] == pytest.approx(expected)
    assert result['probability_floor'] == 1e-12
    assert paired_video_bootstrap(p, q, y)['probability_floor'] == 1e-15


@pytest.mark.parametrize('floor', [0, -1, 1, 2, float('nan'), float('inf')])
def test_invalid_probability_floor(floor):
    p = np.array([[0.5, 0.5]] * 2)
    with pytest.raises(ValueError, match='probability floor'):
        paired_video_bootstrap(p, p, np.array([0, 1]), probability_floor=floor)
