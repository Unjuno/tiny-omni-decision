import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.build_ami_train_sync_review import sample_times, wav_rms_bins


def test_sample_times_are_ordered_and_include_interval_endpoints() -> None:
    assert sample_times(1.0, 2.0, 3) == [1.0, 1.5, 2.0]


@pytest.mark.parametrize("start,end", [(-0.1, 1.0), (1.0, 1.0), (float("nan"), 1.0)])
def test_sample_times_reject_invalid_ranges(start: float, end: float) -> None:
    with pytest.raises(ValueError):
        sample_times(start, end)


def test_wav_rms_bins_normalize_activity_envelope(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    rate = 16000
    quiet = np.zeros(rate // 2, dtype="<i2")
    loud = np.full(rate // 2, 10000, dtype="<i2")
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.concatenate([quiet, loud]).tobytes())

    result = wav_rms_bins(wav_path, 0.0, 1.0, count=4)

    assert result == [0.0, 0.0, 1.0, 1.0]


def test_wav_rms_bins_reject_non_mono_audio(tmp_path: Path) -> None:
    wav_path = tmp_path / "stereo.wav"
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.zeros(1600, dtype="<i2").tobytes())

    with pytest.raises(ValueError, match="mono"):
        wav_rms_bins(wav_path, 0.0, 0.05)


def test_helper_import_does_not_require_media_extras() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "av" or name == "PIL" or name.startswith("PIL."):
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from scripts.build_ami_train_sync_review import sample_times
assert sample_times(0.0, 1.0, 2) == [0.0, 1.0]
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
