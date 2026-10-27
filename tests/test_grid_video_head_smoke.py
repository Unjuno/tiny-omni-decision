import numpy as np
import pytest

pytest.importorskip('torch')
from scripts.smoke_grid_video_head import valid_logits


def test_logit_shape_and_finite_values():
    assert valid_logits(np.zeros((16, 51))).shape == (16, 51)
    with pytest.raises(ValueError, match='invalid'):
        valid_logits(np.full((16, 51), np.inf))
