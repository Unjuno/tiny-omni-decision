import numpy as np
import pytest

from tiny_omni_decision.transcript_likelihood import (
    mean_transcript_logprob,
    rotated_speaker_indices,
)


def test_rotation_is_bijective_and_cross_speaker():
    assert rotated_speaker_indices(["a", "a", "b", "b"]) == [2, 3, 0, 1]
    with pytest.raises(ValueError):
        rotated_speaker_indices(["a", "b", "a", "b"])


def test_excludes_prefix_and_padding_includes_eos():
    # Shifted targets: three prefix tokens, transcript, EOS, padding.
    values = np.array([[-99, -99, -99, -2, -4, -100], [-99, -99, -99, -3, -5, -7]], dtype=float)
    actual = mean_transcript_logprob(values, [6, 7], prefix_length=4)
    np.testing.assert_allclose(actual, [-3, -5])


def test_padding_does_not_change_score():
    x = np.array([[-99, -99, -99, -2, -4]], dtype=float)
    a = mean_transcript_logprob(x, [6], prefix_length=4)
    b = mean_transcript_logprob(np.pad(x, ((0, 0), (0, 3))), [6], prefix_length=4)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("lengths", [[4], [8], [6, 6]])
def test_rejects_invalid_lengths(lengths):
    with pytest.raises(ValueError):
        mean_transcript_logprob(np.zeros((1, 6)), lengths, prefix_length=4)


def test_rejects_nonfinite():
    with pytest.raises(ValueError):
        mean_transcript_logprob(np.full((1, 6), np.nan), [6], prefix_length=4)
