from __future__ import annotations

import hashlib
import json
import wave

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.extract_fsd50k_annotation_features import (  # noqa: E402
    _load_resume,
    atomic_json,
    canonical_sha256,
    decode_mono_pcm_wav,
    encode_full_clip,
    file_sha256,
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


def test_load_resume_revalidates_completed_feature_shard(tmp_path):
    output = tmp_path / "features"
    shards = output / "shards"
    shards.mkdir(parents=True)
    identity = {"fixture": "fixed"}
    ordered_rows = [{"sample_id": "clip-a"}, {"sample_id": "clip-b"}]
    rows = [
        {"feature_index": index, "sample_id": row["sample_id"]}
        for index, row in enumerate(ordered_rows)
    ]
    values = np.ones((2, 384), dtype=np.float32)
    shard = shards / "shard-00000.npz"
    with shard.open("wb") as stream:
        np.savez_compressed(
            stream,
            sample_ids=np.asarray([row["sample_id"] for row in ordered_rows]),
            features=values,
            rows=np.asarray([json.dumps(row) for row in rows]),
        )
    identity_hash = canonical_sha256(identity)
    atomic_json(output / "effective-config.json", {"identity": identity})
    atomic_json(
        output / "progress.json",
        {
            "status": "RUNNING",
            "identity_sha256": identity_hash,
            "completed_assets": 2,
            "shards": [
                {
                    "name": shard.name,
                    "sha256": file_sha256(shard),
                    "first_feature_index": 0,
                    "feature_count": 2,
                }
            ],
        },
    )

    progress, completed, recovered_rows = _load_resume(output, identity, ordered_rows)

    assert progress["status"] == "RUNNING"
    assert completed == 2
    assert recovered_rows == rows


def test_load_resume_rejects_modified_feature_shard(tmp_path):
    output = tmp_path / "features"
    shards = output / "shards"
    shards.mkdir(parents=True)
    identity = {"fixture": "fixed"}
    ordered_rows = [{"sample_id": "clip-a"}]
    shard = shards / "shard-00000.npz"
    values = np.ones((1, 384), dtype=np.float32)
    with shard.open("wb") as stream:
        np.savez_compressed(
            stream,
            sample_ids=np.asarray(["clip-a"]),
            features=values,
            rows=np.asarray([json.dumps({"feature_index": 0, "sample_id": "clip-a"})]),
        )
    identity_hash = canonical_sha256(identity)
    atomic_json(output / "effective-config.json", {"identity": identity})
    atomic_json(
        output / "progress.json",
        {
            "status": "RUNNING",
            "identity_sha256": identity_hash,
            "completed_assets": 1,
            "shards": [
                {
                    "name": shard.name,
                    "sha256": hashlib.sha256(b"wrong").hexdigest(),
                    "first_feature_index": 0,
                    "feature_count": 1,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="failed resume verification"):
        _load_resume(output, identity, ordered_rows)
