import pytest

from tiny_omni_decision.candidate_rotation import align_mean, rotations


def test_each_candidate_visits_each_position_without_target():
    assert rotations(["X", "Y", "Z"]) == [["X", "Y", "Z"], ["Y", "Z", "X"], ["Z", "X", "Y"]]
    with pytest.raises(ValueError):
        rotations(["X", "X"])


def test_identity_alignment_preserves_correct_probabilities():
    assert align_mean(
        ["X", "Y", "Z"], [[0.1, 0.2, 0.7], [0.2, 0.7, 0.1], [0.7, 0.1, 0.2]]
    ) == pytest.approx([0.1, 0.2, 0.7])


def test_constant_position_bias_is_removed():
    assert align_mean(["X", "Y", "Z"], [[0.8, 0.1, 0.1]] * 3) == pytest.approx([1 / 3] * 3)
    with pytest.raises(ValueError):
        align_mean(["X", "Y"], [[0.5, 0.5]])
    with pytest.raises(ValueError):
        align_mean(["X", "Y"], [[float("nan"), 0.5], [0.5, 0.5]])
