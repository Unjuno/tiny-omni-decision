import numpy as np
import pytest

from scripts.audit_clotho_decoded_media import check_waveform


def test_hash_is_stable_and_sensitive():
    a = np.array([0.1, 0.2, -0.4], dtype=np.float32)
    assert check_waveform(a) == check_waveform(a.copy())
    assert check_waveform(a) != check_waveform(a[::-1])


@pytest.mark.parametrize("value", [[], [0, 0], [float("nan")], [float("inf")], [[1, 2]]])
def test_invalid_waveform_rejected(value):
    with pytest.raises(ValueError):
        check_waveform(np.array(value))
