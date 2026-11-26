import numpy as np
import pytest

from scripts.audio_overlap_probe import PreparedWaveformMatcher, peak_normalized_match


def test_matches_original_at_all_valid_offsets_and_lengths():
    rng = np.random.default_rng(17)
    signals = {str(n): rng.normal(size=n) for n in (13, 37, 64, 121)}
    matcher = PreparedWaveformMatcher(signals)
    for left, a in signals.items():
        for right, b in signals.items():
            if len(a) <= len(b):
                expected = peak_normalized_match(a, b)
                actual = matcher.match(left, right)
                assert actual["offset_samples"] == expected["offset_samples"]
                assert actual["absolute_correlation"] == pytest.approx(
                    expected["absolute_correlation"], abs=1e-10
                )


def test_gain_offset_polarity_and_last_offset_preserved():
    rng = np.random.default_rng(17)
    a = rng.normal(size=23)
    b = np.r_[rng.normal(size=47), -2 * a + 5]
    matcher = PreparedWaveformMatcher({"a": a, "b": b})
    result = matcher.match("a", "b")
    assert result["offset_samples"] == 47
    assert result["absolute_correlation"] == pytest.approx(1)


def test_constant_signals_are_undefined_and_wrong_direction_rejected():
    matcher = PreparedWaveformMatcher({"a": np.ones(3), "b": np.ones(7)})
    assert matcher.match("a", "b")["absolute_correlation"] is None
    with pytest.raises(ValueError):
        matcher.match("b", "a")


def test_memory_and_invalid_input_fail_closed():
    with pytest.raises(ValueError, match="budget"):
        PreparedWaveformMatcher({"a": np.ones(100)}, memory_limit_bytes=1)
    with pytest.raises(ValueError):
        PreparedWaveformMatcher({"a": np.array([1, float("nan")])})
