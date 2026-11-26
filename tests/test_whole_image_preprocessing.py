import numpy as np
import pytest

from tiny_omni_decision.whole_image_preprocessing import whole_image_rgb


def test_resize_receives_complete_image_and_fixed_policy():
    image = np.zeros((480, 640, 3), np.uint8)
    image[:, 0, 0] = 255
    image[:, -1, 2] = 255

    def resize(source, size, *, method):
        assert source is image
        assert size == (288, 288) and method == "bilinear"
        output = np.zeros((288, 288, 3), np.float32)
        output[:, 0] = source[0, 0]
        output[:, -1] = source[0, -1]
        return output

    result = whole_image_rgb(image, resize)
    assert result.dtype == np.float32
    assert np.all(result[:, 0, 0] == 1) and np.all(result[:, -1, 2] == 1)


def test_invalid_input_and_backend_outputs_rejected():
    with pytest.raises(ValueError):
        whole_image_rgb(np.zeros((4, 4, 3), np.float32), None)
    with pytest.raises(ValueError):
        whole_image_rgb(np.zeros((4, 4), np.uint8), None)
    with pytest.raises(ValueError):
        whole_image_rgb(np.zeros((4, 4, 3), np.uint8),
                        lambda *a, **k: np.full((288, 288, 3), np.nan))
