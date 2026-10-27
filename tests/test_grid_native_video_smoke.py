import numpy as np
import pytest

pytest.importorskip('av')
from scripts.smoke_grid_native_video import validate_feature


def test_video_feature_shape_and_finiteness():
    assert validate_feature(np.zeros((16, 768))).shape == (16, 768)
    with pytest.raises(ValueError, match='invalid'):
        validate_feature(np.full((16, 768), np.nan))
    with pytest.raises(ValueError, match='invalid'):
        validate_feature(np.zeros((15, 768)))
