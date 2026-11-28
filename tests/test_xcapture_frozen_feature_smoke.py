from __future__ import annotations

import struct

import numpy as np
import pytest

from scripts.extract_xcapture_frozen_feature_smoke import read_float_wav


def _write_wav(path, *, encoding: int, bits: int, payload: bytes) -> None:
    channels = 1
    sample_rate = 16_000
    block_align = channels * bits // 8
    format_chunk = struct.pack(
        "<HHIIHH", encoding, channels, sample_rate, sample_rate * block_align, block_align, bits
    )
    body = b"fmt " + struct.pack("<I", len(format_chunk)) + format_chunk
    body += b"data" + struct.pack("<I", len(payload)) + payload
    riff = b"WAVE" + body
    path.write_bytes(b"RIFF" + struct.pack("<I", len(riff)) + riff)


def test_read_float_wav_decodes_ieee_float32(tmp_path):
    path = tmp_path / "float.wav"
    expected = np.array([-0.25, 0.0, 0.75], dtype="<f4")
    _write_wav(path, encoding=3, bits=32, payload=expected.tobytes())

    actual, sample_rate = read_float_wav(path)

    assert sample_rate == 16_000
    np.testing.assert_array_equal(actual, expected)


def test_read_float_wav_fails_closed_on_pcm(tmp_path):
    path = tmp_path / "pcm.wav"
    _write_wav(path, encoding=1, bits=16, payload=b"\x00\x00")

    with pytest.raises(ValueError, match="expected IEEE-float32"):
        read_float_wav(path)


def test_read_float_wav_rejects_truncated_chunks(tmp_path):
    path = tmp_path / "truncated.wav"
    path.write_bytes(b"RIFF\x10\x00\x00\x00WAVEdata\x20\x00\x00\x00")

    with pytest.raises(ValueError, match="truncated WAV chunk"):
        read_float_wav(path)
