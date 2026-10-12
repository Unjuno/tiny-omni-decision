from fractions import Fraction

import pytest

from tiny_omni_decision.audio_interval import sample_interval


def test_sample_support_rounds_inward():
    assert sample_interval([(0, Fraction(1, 48000), 48000)], 48000, 1, 100001) == (1, 4800)


def test_contiguous_blocks_and_no_padding():
    blocks = [(0, Fraction(1, 48000), 1024), (1024, Fraction(1, 48000), 1024)]
    assert sample_interval(blocks, 48000, 0, 40000) == (0, 1920)
    with pytest.raises(ValueError, match="exceeds"):
        sample_interval(blocks, 48000, 0, 43000)
    with pytest.raises(ValueError, match="gap or overlap"):
        sample_interval(
            [(0, Fraction(1, 48000), 1024), (1025, Fraction(1, 48000), 1024)], 48000, 0, 40000
        )


def test_actual_44100_clock_is_not_assumed_48000():
    assert sample_interval([(0, Fraction(1, 44100), 44100)], 44100, 0, 1000000) == (0, 44100)
