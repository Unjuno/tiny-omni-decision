import numpy as np
import pytest

from scripts.slot_mask_geometry import mask_geometry


def test_known_geometry_and_input_preservation():
    masks = np.zeros((2, 3, 3))
    masks[0, 0, 2] = 1
    masks[1] = 1
    before = masks.copy()
    result = mask_geometry(masks)
    np.testing.assert_allclose(result[0], [1, 0, 0, 0, 1 / 9])
    np.testing.assert_allclose(result[1], [.5, .5, np.sqrt(1 / 6), np.sqrt(1 / 6), 1])
    np.testing.assert_array_equal(masks, before)


@pytest.mark.parametrize('value', [0, -1, 2, float('nan'), float('inf')])
def test_invalid_masks_rejected(value):
    with pytest.raises(ValueError):
        mask_geometry(np.full((1, 2, 2), value))
