import numpy as np
import pytest

from tiny_omni_decision.shared_visual_encoder import SharedVisualEncoder


def test_one_parameter_tree_explicitly_supplied_to_both_profiles():
    calls = []
    variables = {"sentinel": object()}

    class Model:
        def apply(self, weights, frames, **kwargs):
            calls.append((weights, kwargs))
            batch, time = frames.shape[:2]
            x = np.ones((batch, time * 256, 768), dtype=np.float32)
            return x, {"spatial_features": x}

    encoder = SharedVisualEncoder(Model(), variables, compile_fn=lambda fn: fn)
    image = encoder.encode(np.zeros((2, 1, 288, 288, 3), np.float32), modality="image")
    video = encoder.encode(np.zeros((1, 8, 288, 288, 3), np.float32), modality="video")
    assert image.shape == (2, 17, 768) and video.shape == (1, 8, 768)
    assert all(weights is variables for weights, _ in calls)
    assert all(options["train"] is False for _, options in calls)
    assert np.all(image == 1) and np.all(video == 1)


def test_invalid_profiles_fail_before_model_use():
    encoder = SharedVisualEncoder(None, {}, compile_fn=lambda fn: fn)
    with pytest.raises(ValueError):
        encoder.encode(np.zeros((1, 8, 288, 288, 3), np.float32), modality="image")
    with pytest.raises(ValueError):
        encoder.encode(np.full((1, 1, 288, 288, 3), np.nan, np.float32), modality="image")
    with pytest.raises(ValueError):
        encoder.encode(np.zeros((1, 1, 288, 288, 3), np.float32), modality="audio")


def test_wrong_output_batch_rejected():
    class BadModel:
        def apply(self, *args, **kwargs):
            return None, {"spatial_features": np.zeros((2, 256, 768), np.float32)}

    encoder = SharedVisualEncoder(BadModel(), {}, compile_fn=lambda fn: fn)
    with pytest.raises(ValueError, match="batch"):
        encoder.encode(np.zeros((1, 1, 288, 288, 3), np.float32), modality="image")
