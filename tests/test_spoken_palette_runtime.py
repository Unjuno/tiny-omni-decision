from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from tiny_omni_decision.decision import FrozenFeatureCandidateScorer
from tiny_omni_decision.spoken_palette_runtime import (
    SpokenPaletteRuntime,
    _checked_bundle_file,
    read_pcm_wav,
    waveform_from_wav_bytes,
)


def _wav_bytes(*, rate: int = 16000, channels: int = 1, width: int = 2) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(np.asarray([0, 1000, -1000], dtype="<i2").tobytes())
    return buffer.getvalue()


def test_pcm_wav_readers_accept_exact_runtime_format(tmp_path: Path) -> None:
    content = _wav_bytes()
    path = tmp_path / "voice.wav"
    path.write_bytes(content)
    expected = np.asarray([0.0, 1000 / 32768.0, -1000 / 32768.0], dtype=np.float32)
    np.testing.assert_array_equal(read_pcm_wav(path), expected)
    np.testing.assert_array_equal(waveform_from_wav_bytes(content), expected)


@pytest.mark.parametrize(
    ("rate", "channels", "width"), [(8000, 1, 2), (16000, 2, 2), (16000, 1, 1)]
)
def test_pcm_wav_readers_reject_unapproved_formats(
    tmp_path: Path, rate: int, channels: int, width: int
) -> None:
    content = _wav_bytes(rate=rate, channels=channels, width=width)
    path = tmp_path / "voice.wav"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="mono 16 kHz"):
        read_pcm_wav(path)
    with pytest.raises(ValueError, match="mono 16 kHz"):
        waveform_from_wav_bytes(content)


def test_float32_pcm_wav_reader_accepts_source_format() -> None:
    samples = np.asarray([0.0, 0.25, -0.5, 0.75], dtype="<f4")
    payload = samples.tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(payload)) + payload
    content = b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
    np.testing.assert_array_equal(waveform_from_wav_bytes(content), samples)


def test_bundle_path_and_sha_are_checked(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    payload = root / "weights.bin"
    payload.write_bytes(b"fixed")
    import hashlib

    expected = hashlib.sha256(b"fixed").hexdigest()
    assert _checked_bundle_file(root, "weights.bin", expected) == payload.resolve()
    with pytest.raises(ValueError, match="hash mismatch"):
        _checked_bundle_file(root, "weights.bin", "0" * 64)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"fixed")
    with pytest.raises(ValueError, match="escapes"):
        _checked_bundle_file(root, "../outside.bin", expected)


def test_json_decision_preserves_supplied_option_order_and_probabilities() -> None:
    runtime = object.__new__(SpokenPaletteRuntime)
    runtime.mean = np.zeros(1538, dtype=np.float32)
    runtime.std = np.ones(1538, dtype=np.float32)
    runtime.temperature = 1.0
    runtime.scorer = FrozenFeatureCandidateScorer(1538, hidden_size=64).eval()
    runtime._encode_audio = lambda _waveform: np.ones(384, dtype=np.float32)
    runtime._encode_visual_options = lambda _path, positions: np.zeros(
        (len(positions), 1154), dtype=np.float32
    )
    options = [
        {"id": "opaque-z", "text": "candidate Z", "x_norm": 0.75, "y_norm": 0.68},
        {"id": "opaque-a", "text": "candidate A", "x_norm": 0.25, "y_norm": 0.34},
    ]

    result = json.loads(
        runtime.decide_waveform(
            np.zeros(320, dtype=np.float32),
            Path("unused-by-mocked-visual-encoder.png"),
            request_id="runtime-test",
            options=options,
        )
    )
    assert [item["id"] for item in result["options"]] == ["opaque-z", "opaque-a"]
    assert sum(item["probability"] for item in result["options"]) == pytest.approx(1.0)
    assert result["selected_option_id"] in {"opaque-z", "opaque-a"}


def test_options_reject_duplicate_ids_and_bad_coordinates() -> None:
    duplicate = [
        {"id": "same", "text": "first", "x_norm": 0.5, "y_norm": 0.5},
        {"id": "same", "text": "second", "x_norm": 0.2, "y_norm": 0.8},
    ]
    with pytest.raises(ValueError, match="unique"):
        SpokenPaletteRuntime._option_positions(duplicate)
    duplicate[1]["id"] = "other"
    duplicate[1]["x_norm"] = float("nan")
    with pytest.raises(ValueError, match="coordinates"):
        SpokenPaletteRuntime._option_positions(duplicate)
