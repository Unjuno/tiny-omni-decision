import numpy as np
import pytest

from scripts.extract_starss23_foa_channel_features import encode_foa_window


def test_foa_feature_extraction_preserves_channel_order():
    audio = np.zeros((120000, 4), dtype=np.float32)
    audio[:, 0] = 1
    audio[:, 1] = 2
    audio[:, 2] = 3
    audio[:, 3] = 4

    features = encode_foa_window(
        audio,
        lambda data: np.full(384, data["audio"].mean(), dtype=np.float32),
    )

    np.testing.assert_array_equal(features[:, 0], [1, 2, 3, 4])


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros((120000, 3), dtype=np.float32),
        np.full((120000, 4), np.nan, dtype=np.float32),
    ],
)
def test_foa_feature_extraction_rejects_invalid_audio(audio):
    with pytest.raises(ValueError, match="four-channel"):
        encode_foa_window(audio, lambda data: np.zeros(384, dtype=np.float32))
