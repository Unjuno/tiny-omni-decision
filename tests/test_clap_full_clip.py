import numpy as np
import pytest

from scripts.clap_full_clip import duration_weighted_feature, window_spans


@pytest.mark.parametrize("length", [1, 10, 20, 21, 37])
def test_every_sample_is_covered_exactly_once(length):
    spans = window_spans(length, 10)
    rebuilt = np.concatenate([np.arange(start, stop) for start, stop in spans])
    np.testing.assert_array_equal(rebuilt, np.arange(length))
    assert all(0 < stop - start <= 10 for start, stop in spans)
    assert len(spans) == (length + 9) // 10


def test_partial_tail_is_duration_weighted():
    pooled = duration_weighted_feature(np.eye(2), [10, 1])
    np.testing.assert_allclose(pooled, np.array([10, 1]) / np.sqrt(101))
    assert not np.allclose(pooled, np.ones(2) / np.sqrt(2))


@pytest.mark.parametrize("length,maximum", [(0, 10), (-1, 10), (10, 0), (10.5, 10)])
def test_invalid_coverage_arguments_rejected(length, maximum):
    with pytest.raises(ValueError):
        window_spans(length, maximum)


@pytest.mark.parametrize(
    "features,weights",
    [
        (np.eye(2), [1]),
        (np.eye(2), [1, 0]),
        (np.array([[np.nan, 1]]), [1]),
        (np.zeros((1, 2)), [1]),
    ],
)
def test_bad_pooling_input_rejected(features, weights):
    with pytest.raises(ValueError):
        duration_weighted_feature(features, weights)
