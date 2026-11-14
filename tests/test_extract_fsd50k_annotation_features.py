from __future__ import annotations

import wave

import numpy as np
import pytest
import torch

from scripts.extract_fsd50k_annotation_features import (
    decode_mono_pcm_wav,
    encode_full_clip,
)


def _write_pcm(path, values, *, channels=1, sample_rate=44_100):
    interleaved = np.asarray(values, dtype="<i2").reshape(-1)
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(interleaved.tobytes())


def test_decode_mono_pcm_wav_preserves_full_clip_and_rate(tmp_path):
    path = tmp_path / "audio.wav"
    values = np.asarray([0, -32768, 32767, 16384], dtype=np.int16)
    _write_pcm(path, values)

    waveform, sample_rate = decode_mono_pcm_wav(path)

    assert sample_rate == 44_100
    assert waveform.dtype == np.float32
    assert waveform.shape == values.shape
    assert waveform[1] == -1.0
    assert waveform[2] == pytest.approx(32767 / 32768)


def test_decode_mono_pcm_wav_rejects_stereo_input(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_pcm(path, [[1, 2], [3, 4]], channels=2)
    with pytest.raises(ValueError, match="mono PCM"):
        decode_mono_pcm_wav(path)


class FakeMel:
    def __init__(self):
        self.input_samples = None

    def __call__(self, waveform):
        self.input_samples = waveform.shape[-1]
        return waveform.unsqueeze(1)


class FakeEncoder:
    device = torch.device("cpu")

    def __init__(self):
        self.mel = FakeMel()
        self.model_input_shape = None

    def model(self, values):
        self.model_input_shape = tuple(values.shape)
        return torch.zeros((1, 2)), torch.ones((1, 384))


def test_encode_full_clip_uses_whole_mono_waveform_without_windowing():
    encoder = FakeEncoder()
    waveform = np.linspace(-0.5, 0.5, 32_000, dtype=np.float32)

    features, resampled_samples = encode_full_clip(
        waveform, 32_000, encoder=encoder, torchaudio_module=None
    )

    assert features.shape == (384,)
    assert np.all(features == 1.0)
    assert resampled_samples == len(waveform)
    assert encoder.mel.input_samples == len(waveform)
    assert encoder.model_input_shape == (1, 1, 1, len(waveform))


def test_encode_full_clip_resamples_native_rate_before_encoding():
    class Functional:
        @staticmethod
        def resample(values, source_rate, target_rate):
            assert source_rate == 44_100
            assert target_rate == 32_000
            size = round(values.shape[-1] * target_rate / source_rate)
            return torch.nn.functional.interpolate(
                values.unsqueeze(1), size=size, mode="linear", align_corners=False
            ).squeeze(1)

    class Audio:
        functional = Functional()

    encoder = FakeEncoder()
    waveform = np.zeros(44_100, dtype=np.float32)

    _features, resampled_samples = encode_full_clip(
        waveform, 44_100, encoder=encoder, torchaudio_module=Audio
    )

    assert resampled_samples == 32_000
    assert encoder.mel.input_samples == 32_000
