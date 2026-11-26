import numpy as np
import pytest

from scripts.probe_robomnist_av_match_features import _pcm_to_float


def test_pcm_to_float_decodes_signed_16_bit_samples():
    payload = np.array([-32768, 0, 32767], dtype="<i2").tobytes()

    result = _pcm_to_float(payload, 2)

    assert result[0] == -1.0
    assert result[1] == 0.0
    assert result[2] == pytest.approx(32767 / 32768)


def test_pcm_to_float_decodes_signed_24_bit_samples():
    payload = bytes((0x00, 0x00, 0x80, 0xFF, 0xFF, 0x7F))

    result = _pcm_to_float(payload, 3)

    assert result[0] == -1.0
    assert result[1] == pytest.approx(8_388_607 / 8_388_608)


def test_pcm_to_float_rejects_misaligned_24_bit_payload():
    with pytest.raises(ValueError, match="divisible by three"):
        _pcm_to_float(b"\x00\x01", 3)
