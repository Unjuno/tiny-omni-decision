import pytest

from scripts.audit_vjepa21_checkpoint import expected_patch_tokens, uniform_frame_indices


def test_uniform_frame_indices_are_even_and_stable():
    assert uniform_frame_indices(9, 5) == (0, 2, 4, 6, 8)
    assert uniform_frame_indices(4, 8) == (0, 0, 1, 1, 2, 2, 3, 3)
    assert uniform_frame_indices(1, 4) == (0, 0, 0, 0)


@pytest.mark.parametrize("frame_count,requested_count", [(0, 1), (1, 0), (-2, 4)])
def test_uniform_frame_indices_reject_invalid_counts(frame_count, requested_count):
    with pytest.raises(ValueError, match="must be positive"):
        uniform_frame_indices(frame_count, requested_count)


def test_vjepa21_image_and_video_patch_token_counts():
    assert expected_patch_tokens(1, 384, 16, 1) == 576
    assert expected_patch_tokens(8, 384, 16, 2) == 2304


def test_patch_token_count_rejects_non_divisible_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        expected_patch_tokens(7, 384, 16, 2)
