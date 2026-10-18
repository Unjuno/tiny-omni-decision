import numpy as np
import pytest

from tiny_omni_decision.mirror_augmentation import mirror_mask


def test_mirror_mask_deterministic_image_grouping():
    a = mirror_mask(8424, 1)
    np.testing.assert_array_equal(a, mirror_mask(8424, 1))
    assert a.dtype == np.bool_ and 0 < a.sum() < len(a)
    assert not np.array_equal(a, mirror_mask(8424, 2))
    question_ids = np.arange(8424 * 4)
    grouped = a[question_ids // 4].reshape(-1, 4)
    assert np.all(grouped == a[:, None])
    with pytest.raises(ValueError):
        mirror_mask(0, 1)
