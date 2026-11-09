from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_path_a_videomaev2_roundtrip import (
    normalize_rgb_frames,
    static_image_clip,
)


def test_image_route_repeats_exact_static_frame_for_16_frame_encoder() -> None:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[10, 20] = [255, 127, 0]

    frames = static_image_clip(image)

    assert frames.shape == (16, 224, 224, 3)
    assert all(np.array_equal(frame, image) for frame in frames)


def test_video_preprocessing_matches_expected_channel_time_layout_and_range() -> None:
    frames = np.zeros((16, 224, 224, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 255

    values = normalize_rgb_frames(frames)

    assert values.shape == (1, 3, 16, 224, 224)
    assert values.dtype == np.float32
    assert values[0, 0].min() == values[0, 0].max() == 1.0
    assert values[0, 1].min() == values[0, 1].max() == -1.0
    assert values[0, 2].min() == values[0, 2].max() == -1.0


def test_image_preprocessing_rejects_wrong_shape_and_dtype() -> None:
    with pytest.raises(ValueError, match="uint8 RGB"):
        static_image_clip(np.zeros((224, 224, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        normalize_rgb_frames(np.zeros((8, 224, 224, 3), dtype=np.uint8))
