import numpy as np
import pytest

from tiny_omni_decision.foa_direction import foa_direction


def test_known_foa_channel_order_and_scale_invariance():
    signal = np.sin(np.arange(2400) * .1)
    # Documented W,Y,Z,X response for azimuth+90,elevation0.
    audio = np.stack([signal, signal, signal * 0, signal * 0], -1)
    vector = foa_direction(audio)
    np.testing.assert_allclose(vector, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(foa_direction(audio * .1), vector, atol=1e-12)
    audio = np.stack([signal, signal * 0, signal * 0, signal], -1)
    np.testing.assert_allclose(foa_direction(audio), [1, 0, 0], atol=1e-12)


def test_no_direction_and_bad_inputs():
    assert foa_direction(np.zeros((2400, 4))) is None
    for invalid in (np.zeros((4, 3)), np.zeros((0, 4)), np.full((5, 4), np.nan)):
        with pytest.raises(ValueError):
            foa_direction(invalid)
