import numpy as np
import pytest

from tiny_omni_decision.dinov2_features import cls_patch_mean, require_image_splits


def test_cls_patch_mean_separates_cls():
    x = np.ones((2, 257, 384), dtype=np.float32)
    x[:, 0] = 5
    out = cls_patch_mean(x)
    assert out.shape == (2, 768)
    np.testing.assert_array_equal(out[:, :384], 5)
    np.testing.assert_array_equal(out[:, 384:], 1)


def test_invalid_features_fail():
    for x in [np.zeros((2, 256, 384)), np.full((1, 257, 384), np.nan)]:
        with pytest.raises(ValueError):
            cls_patch_mean(x)


def test_split_identity_rejections():
    require_image_splits([{"id": "a", "sha256": "x"}], [{"id": "b", "sha256": "y"}])
    for other in [{"id": "a", "sha256": "y"}, {"id": "b", "sha256": "x"}]:
        with pytest.raises(ValueError):
            require_image_splits([{"id": "a", "sha256": "x"}], [other])
    with pytest.raises(ValueError):
        require_image_splits([{"id": "a", "sha256": "x"}] * 2, [{"id": "b", "sha256": "y"}])
