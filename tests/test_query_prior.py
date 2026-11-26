import pytest

from tiny_omni_decision.query_prior import leave_group_out_prior


def test_excludes_whole_video_and_smooths_unseen_class():
    probabilities = leave_group_out_prior(
        ["a", "a", "a", "b"], ["v1", "v1", "v2", "v3"], [1, 1, 0, 1]
    )
    assert probabilities[:2] == [[2 / 3, 1 / 3]] * 2
    assert probabilities[2] == [1 / 4, 3 / 4]
    assert probabilities[3] == [0.5, 0.5]
    with pytest.raises(ValueError):
        leave_group_out_prior(["a"], ["v"], [2])
