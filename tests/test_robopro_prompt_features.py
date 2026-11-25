import numpy as np
import pytest

from scripts.extract_robopro_prompt_features import masked_mean_pool


def test_masked_mean_pool_ignores_padding_and_preserves_unscaled_mean():
    hidden = np.asarray(
        [[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]],
        dtype=np.float32,
    )
    mask = np.asarray([[1, 1, 0]], dtype=np.int64)

    result = masked_mean_pool(hidden, mask)

    np.testing.assert_allclose(result, [[2.0, 4.0]])


def test_masked_mean_pool_rejects_empty_or_invalid_batches():
    with pytest.raises(ValueError, match="at least one unmasked"):
        masked_mean_pool(np.zeros((1, 2, 3)), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="expected hidden"):
        masked_mean_pool(np.zeros((2, 3)), np.ones((2, 3)))
