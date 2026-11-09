from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from tiny_omni_decision.textfree_audio import TextFreeAudioReadout  # noqa: E402
from tiny_omni_decision.textfree_audio_runtime import (  # noqa: E402
    SPEECH_COMMANDS_CLASSES,
    TextFreeAudioRuntime,
)


def _write_wav(path: Path, *, sample_rate: int = 16_000, samples: int = 16_000) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes((torch.zeros(samples, dtype=torch.int16)).numpy().tobytes())


@pytest.fixture
def runtime_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import tiny_omni_decision.textfree_audio_runtime as module

    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    config_path = encoder_dir / "config.json"
    weights_path = encoder_dir / "encoder.safetensors"
    processor_path = encoder_dir / "preprocessor_config.json"
    readout_path = tmp_path / "readout.pt"
    config_path.write_text("{}", encoding="utf-8")
    weights_path.write_bytes(b"fake-encoder")
    processor_path.write_text("{}", encoding="utf-8")

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class FakeEncoder(torch.nn.Module):
        def __init__(self, _config):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.calls = 0

        def forward(self, input_features):
            self.calls += 1
            self.autocast_enabled = torch.is_autocast_enabled(input_features.device.type)
            self.autocast_dtype = torch.get_autocast_dtype(input_features.device.type)
            batch = input_features.shape[0]
            hidden = torch.ones((batch, 1500, 384), device=input_features.device)
            return SimpleNamespace(last_hidden_state=hidden * self.scale)

    class FakeFeatureExtractor:
        n_samples = 480_000
        hop_length = 160

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, waveform, **_kwargs):
            assert len(waveform) == 16_000
            return SimpleNamespace(input_features=torch.zeros((1, 80, 3000)))

    monkeypatch.setattr(module, "WhisperConfig", FakeConfig)
    monkeypatch.setattr(module, "WhisperEncoder", FakeEncoder)
    monkeypatch.setattr(module, "WhisperFeatureExtractor", FakeFeatureExtractor)
    monkeypatch.setattr(
        module, "load_file", lambda *_args, **_kwargs: {"scale": torch.ones(())}
    )
    monkeypatch.setattr(module, "AUDIO_ENCODER_PARAMETERS", 1)

    readout = TextFreeAudioReadout(384, list(SPEECH_COMMANDS_CLASSES)).eval()
    torch.manual_seed(17)
    torch.save(readout.state_dict(), readout_path)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(module, "AUDIO_ENCODER_CONFIG_SHA256", digest(config_path))
    monkeypatch.setattr(module, "AUDIO_ENCODER_SHA256", digest(weights_path))
    monkeypatch.setattr(module, "AUDIO_PREPROCESSOR_SHA256", digest(processor_path))
    monkeypatch.setattr(module, "AUDIO_READOUT_SHA256", digest(readout_path))

    def create(
        cache_entries: int = 4,
        expected_readout_sha256: str | None = None,
    ) -> TextFreeAudioRuntime:
        return TextFreeAudioRuntime(
            encoder_dir,
            readout_path,
            device="cpu",
            cache_entries=cache_entries,
            expected_readout_sha256=expected_readout_sha256,
        )

    create.readout_sha256 = digest(readout_path)
    return create


def _options(classes: tuple[str, ...] = SPEECH_COMMANDS_CLASSES):
    return [
        {"option_id": f"caller-{class_id}", "class_id": class_id}
        for class_id in classes
    ]


def test_raw_audio_decision_reuses_features_across_option_order_and_subset(
    runtime_factory, tmp_path: Path
) -> None:
    audio_path = tmp_path / "train.wav"
    _write_wav(audio_path)
    runtime = runtime_factory()

    first = runtime.decide(audio_path, _options())
    reordered_classes = tuple(reversed(SPEECH_COMMANDS_CLASSES))
    reordered = runtime.decide(audio_path, _options(reordered_classes))
    subset = runtime.decide(audio_path, _options(("yes", "no")))

    first_scores = {row["option_id"]: row["probability"] for row in first["options"]}
    reordered_scores = {row["option_id"]: row["probability"] for row in reordered["options"]}
    assert first["selected_option_id"] == reordered["selected_option_id"]
    assert first_scores == pytest.approx(reordered_scores)
    assert len(subset["options"]) == 2
    assert sum(row["probability"] for row in subset["options"]) == pytest.approx(1.0)
    assert runtime.encoder.calls == 1
    assert runtime.encoder.autocast_enabled is True
    assert runtime.encoder.autocast_dtype is torch.bfloat16
    assert first["encoder_metrics"]["cache_hit"] is False
    assert reordered["encoder_metrics"]["cache_hit"] is True
    assert subset["encoder_metrics"]["single_event_encoder_calls"] == 1
    runtime.close()


def test_raw_audio_runtime_rejects_hash_mismatch_before_encoding(runtime_factory, tmp_path):
    audio_path = tmp_path / "train.wav"
    _write_wav(audio_path)
    runtime = runtime_factory()

    with pytest.raises(ValueError, match="audio hash differs"):
        runtime.decide(audio_path, _options(), expected_media_sha256="0" * 64)

    assert runtime.encoder.calls == 0
    assert runtime.audio_cache is not None and len(runtime.audio_cache) == 0
    runtime.close()


def test_runtime_accepts_explicit_hash_for_frozen_alternate_readout(runtime_factory):
    runtime = runtime_factory(
        expected_readout_sha256=runtime_factory.readout_sha256
    )
    runtime.close()


def test_runtime_rejects_unpinned_alternate_readout_hash(runtime_factory):
    with pytest.raises(ValueError, match="readout.pt"):
        runtime_factory(expected_readout_sha256="0" * 64)


@pytest.mark.parametrize("digest", ["", "A" * 64, "z" * 64])
def test_runtime_rejects_malformed_explicit_readout_hash(runtime_factory, digest):
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        runtime_factory(expected_readout_sha256=digest)


def test_raw_audio_runtime_requires_exact_sample_format(runtime_factory, tmp_path):
    audio_path = tmp_path / "wrong-rate.wav"
    _write_wav(audio_path, sample_rate=8_000)
    runtime = runtime_factory()

    with pytest.raises(ValueError, match="16 kHz PCM WAV"):
        runtime.decide(audio_path, _options())

    assert runtime.encoder.calls == 0
    runtime.close()
