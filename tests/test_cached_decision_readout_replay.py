from __future__ import annotations

import numpy as np
import pytest

from scripts.replay_cached_decision_readout import pad_score_rows


def test_pad_score_rows_masks_only_unsupplied_options():
    scores = pad_score_rows([np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])])

    assert scores.shape == (2, 3)
    assert np.array_equal(scores[0, :2], [1.0, 2.0])
    assert scores[0, 2] < -1e20
    assert np.array_equal(scores[1], [3.0, 4.0, 5.0])


@pytest.mark.parametrize(
    "rows",
    [[], [np.array([1.0])], [np.array([1.0, float("nan")])], [np.ones((2, 2))]],
)
def test_pad_score_rows_rejects_invalid_rows(rows):
    with pytest.raises(ValueError):
        pad_score_rows(rows)
