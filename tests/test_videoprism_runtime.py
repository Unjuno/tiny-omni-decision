from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
videoprism_runtime = pytest.importorskip("tiny_omni_decision.videoprism_runtime")
candidate_features = videoprism_runtime.candidate_features


def test_candidate_features_preserves_image_and_option_order() -> None:
    image = np.full((768,), 2.0, dtype=np.float32)
    options = np.stack([np.full((384,), 3.0), np.full((384,), 4.0)])

    features = candidate_features(image, options)

    assert features.shape == (2, 1152)
    np.testing.assert_array_equal(features[:, :768], np.full((2, 768), 2.0))
    np.testing.assert_array_equal(features[:, 768:], options)


@pytest.mark.parametrize(
    ("image", "options", "message"),
    [
        (np.zeros((4,), dtype=np.float32), np.zeros((2, 384), dtype=np.float32), "image features"),
        (
            np.zeros((768,), dtype=np.float32),
            np.zeros((1, 384), dtype=np.float32),
            "option embeddings",
        ),
        (np.zeros((768,), dtype=np.float32), np.full((2, 384), np.nan), "finite"),
    ],
)
def test_candidate_features_rejects_invalid_inputs(image, options, message) -> None:
    with pytest.raises(ValueError, match=message):
        candidate_features(image, options)
