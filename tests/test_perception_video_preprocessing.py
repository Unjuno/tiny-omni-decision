import numpy as np
import pytest

from tiny_omni_decision.perception_video_preprocessing import letterbox_videoprism_frames


def test_letterbox_preserves_wide_frame_and_centers_padding():
    frame = np.full((2, 4, 3), 255, dtype=np.uint8)
    result = letterbox_videoprism_frames([frame], lambda image, _size: image, frame_size=4)

    assert result.shape == (1, 4, 4, 3)
    assert np.all(result[:, 0] == 0)
    assert np.all(result[:, 3] == 0)
    assert np.all(result[:, 1:3] == 1)


def test_letterbox_rejects_bad_frames_and_resize_shape():
    with pytest.raises(ValueError, match="finite nonempty"):
        letterbox_videoprism_frames([], lambda image, _size: image)
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="unexpected frame shape"):
        letterbox_videoprism_frames([frame], lambda _image, _size: frame[:1])
