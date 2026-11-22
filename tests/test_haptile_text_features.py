from __future__ import annotations

import numpy as np
import pytest

from scripts.pretrained_reuse.extract_haptile_text_features import pair_features


def test_pair_features_are_question_option_interactions() -> None:
    question = np.ones((2, 384), dtype=np.float32)
    options = np.zeros((7, 384), dtype=np.float32)
    actual = pair_features(question, options)
    assert actual.shape == (2, 7, 1536)
    np.testing.assert_array_equal(actual[..., :384], 1)
    np.testing.assert_array_equal(actual[..., 384:768], 0)
    np.testing.assert_array_equal(actual[..., 768:1152], 1)
    np.testing.assert_array_equal(actual[..., 1152:], 0)


@pytest.mark.parametrize(
    "question,options",
    [
        (np.zeros((2, 10)), np.zeros((7, 384))),
        (np.full((1, 384), np.nan), np.zeros((7, 384))),
        (np.zeros((1, 384)), np.zeros((6, 384))),
    ],
)
def test_pair_features_reject_bad_encoder_outputs(question, options) -> None:
    with pytest.raises(ValueError, match="finite 384-wide"):
        pair_features(question, options)
