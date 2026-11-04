import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tiny_omni_decision.robomnist_temporal_cv import (  # noqa: E402
    TemporalAVMatchScorer,
    normalize_sequences,
    permute_by_group,
)


def test_normalizer_excludes_held_source():
    features = {
        "fit": (np.zeros((8, 2), np.float32), np.zeros((8, 3), np.float32)),
        "held": (np.full((8, 2), 1e5, np.float32), np.full((8, 3), -1e5, np.float32)),
    }
    _, stats = normalize_sequences(features, {"fit"})
    assert np.array_equal(stats["audio_mean"], np.zeros(2))
    assert np.array_equal(stats["video_mean"], np.zeros(3))
    assert np.array_equal(stats["audio_scale"], np.ones(2))


def test_group_shuffles_are_deterministic_and_independent_by_stream():
    x = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    features = {"shared": (x, x.copy()), "other": (x.copy(), x.copy())}
    first = permute_by_group(features, seed=17)
    again = permute_by_group(features, seed=17)
    for group in features:
        for a, b in zip(first[group], again[group], strict=True):
            assert np.array_equal(a, b)
            assert set(map(tuple, a.tolist())) == set(map(tuple, x.tolist()))
    assert any(not np.array_equal(first["shared"][i], features["shared"][i]) for i in (0, 1))


def test_lstm_shape_finiteness_parameter_budget_and_reload():
    torch.manual_seed(17)
    model = TemporalAVMatchScorer()
    assert model.trainable_parameters == 76032
    audio, video = torch.randn(3, 8, 384), torch.randn(3, 8, 768)
    logits = model(audio, video)
    assert logits.shape == (3, 2) and torch.isfinite(logits).all()
    clone = TemporalAVMatchScorer()
    clone.load_state_dict(model.state_dict())
    assert torch.equal(logits, clone(audio, video))
    reverse = model(audio.flip(1), video.flip(1))
    assert not torch.equal(logits, reverse)


def test_bad_sequence_shape_and_nonfinite_fail_closed():
    model = TemporalAVMatchScorer()
    with pytest.raises(ValueError):
        model(torch.zeros(2, 7, 384), torch.zeros(2, 8, 768))
    audio = torch.zeros(2, 8, 384)
    audio[0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        model(audio, torch.zeros(2, 8, 768))
