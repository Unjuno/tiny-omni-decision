import numpy as np
import pytest

from scripts.audio_overlap_probe import peak_normalized_match


def test_shift_gain_and_polarity():
    rng = np.random.default_rng(17)
    a, b = rng.normal(size=100), rng.normal(size=500)
    b[123:223] = -3 * a + 0.4
    result = peak_normalized_match(a, b)
    assert result["offset_samples"] == 123
    assert result["absolute_correlation"] == pytest.approx(1)


def test_fft_matches_direct_reference():
    rng = np.random.default_rng(18)
    a, b = rng.normal(size=11), rng.normal(size=67)
    direct = [abs(np.corrcoef(a, b[i:i + len(a)])[0, 1])
              for i in range(len(b) - len(a) + 1)]
    result = peak_normalized_match(a, b)
    assert result["offset_samples"] == np.argmax(direct)
    assert result["absolute_correlation"] == pytest.approx(max(direct))


def test_silence_is_unknown_not_match():
    assert peak_normalized_match(np.zeros(5), np.ones(10))["offset_samples"] is None
    assert peak_normalized_match(np.arange(5), np.ones(10))["offset_samples"] is None


def test_invalid_input():
    with pytest.raises(ValueError):
        peak_normalized_match([1, float("nan")], [1, 2, 3])
    with pytest.raises(ValueError):
        peak_normalized_match([1, 2, 3], [1, 2])
