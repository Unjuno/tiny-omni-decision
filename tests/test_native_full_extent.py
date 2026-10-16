"""Full-canvas preprocessing contracts without importing the optional encoder stack."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.native_full_extent import full_extent_geometry, preprocess_full_extent


@pytest.mark.parametrize(
    "height,width,expected",
    [
        (480, 640, (216, 288, 36, 0)),
        (320, 480, (192, 288, 48, 0)),
        (640, 480, (288, 216, 0, 36)),
        (288, 288, (288, 288, 0, 0)),
    ],
)
def test_geometry_retains_full_source_extent(height, width, expected):
    result = full_extent_geometry(height, width)
    assert (
        result["resized_height"],
        result["resized_width"],
        result["pad_top"],
        result["pad_left"],
    ) == expected
    assert result["source_box"] == [0, 0, width, height]
    assert result["resized_height"] + result["pad_top"] + result["pad_bottom"] == 288
    assert result["resized_width"] + result["pad_left"] + result["pad_right"] == 288
    assert result["fill_rgb"] == [0, 0, 0]


@pytest.mark.parametrize("shape", [(0, 10), (10, -1), (True, 10), (10.0, 20)])
def test_geometry_rejects_invalid_dimensions(shape):
    with pytest.raises(ValueError, match="positive integer"):
        full_extent_geometry(*shape)


@pytest.mark.parametrize("frames", [None, 2])
def test_all_four_corners_survive_and_padding_is_separate(frames):
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[0, 0], image[0, -1] = [255, 0, 0], [0, 255, 0]
    image[-1, 0], image[-1, -1] = [0, 0, 255], [255, 255, 255]
    value = image if frames is None else np.stack([image, image // 2])
    original = value.copy()

    def resize(data, shape, method):
        assert shape == (4, 8) and method == "bilinear"
        return np.asarray(data, dtype=np.float32)

    tf = SimpleNamespace(image=SimpleNamespace(resize=resize))
    result, geometry = preprocess_full_extent(value, tf, size=8)
    assert result.dtype == np.float32
    assert result.shape == ((8, 8, 3) if frames is None else (2, 8, 8, 3))
    np.testing.assert_array_equal(result[..., 2:6, :, :], original.astype(np.float32) / 255)
    assert np.all(result[..., :2, :, :] == 0) and np.all(result[..., 6:, :, :] == 0)
    np.testing.assert_array_equal(value, original)
    assert geometry["source_box"] == [0, 0, 8, 4]


@pytest.mark.parametrize(
    "value", [np.zeros((4, 8)), np.zeros((4, 8, 4), np.uint8), np.zeros((4, 8, 3), np.float32)]
)
def test_preprocess_requires_original_rgb_uint8(value):
    with pytest.raises(ValueError, match="RGB uint8"):
        preprocess_full_extent(value, None)
