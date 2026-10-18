import pytest

from scripts.prepare_greatest_hits_av_match_inputs import uniform_frame_indices


def test_uniform_frame_indices_include_both_endpoints_and_are_deterministic():
    indices = uniform_frame_indices(101)

    assert indices == [0, 14, 29, 43, 57, 71, 86, 100]
    assert indices == uniform_frame_indices(101)


def test_uniform_frame_indices_reject_short_videos():
    with pytest.raises(ValueError, match="at least eight"):
        uniform_frame_indices(7)
