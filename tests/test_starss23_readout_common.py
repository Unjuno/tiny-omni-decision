import pytest

from scripts.starss23_readout_common import different_recording_permutation


def test_video_control_preserves_marginal_and_changes_every_recording():
    recordings = ["a"] * 3 + ["b"] * 2 + ["c"] * 2
    order = different_recording_permutation(recordings)
    assert sorted(order) == list(range(7))
    assert all(recordings[i] != recordings[j] for i, j in enumerate(order))
    assert order == different_recording_permutation(recordings)


def test_impossible_control_fails_closed():
    with pytest.raises(ValueError):
        different_recording_permutation(["a", "a", "a", "b"])


def test_binary_metrics_brier_and_absent_class():
    pytest.importorskip("numpy")
    from scripts.starss23_readout_common import binary_metrics

    result = binary_metrics([0, 1], [0.5, 0.5])
    assert result["accuracy"] == 0.5
    assert result["brier"] == 0.5
    assert result["ece"] == 0
    assert sum(result["ece_bin_counts"]) == 2
    assert binary_metrics([0, 0], [0, 0])["balanced_accuracy"] is None


def test_invalid_probabilities_fail():
    pytest.importorskip("numpy")
    from scripts.starss23_readout_common import binary_metrics

    with pytest.raises(ValueError):
        binary_metrics([1], [float("nan")])
