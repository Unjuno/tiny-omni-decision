"""CPU-capable raw-audio runtime for the frozen, text-free command readout.

This is a narrow closed-taxonomy Speech Commands route. It loads only the
encoder weights, never the Whisper decoder or a text encoder. It does not
transcribe audio or accept question/candidate text.
"""

from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from tiny_omni_decision.feature_cache import BoundedFeatureCache
from tiny_omni_decision.textfree_audio import TextFreeAudioReadout

AUDIO_ENCODER_REVISION = "openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
AUDIO_ENCODER_SHA256 = "d7f4b75d10f8ec4af79a32cc6dd20160f0c669c67bd104c1f1ab3f9d5d36a2a8"
AUDIO_ENCODER_PARAMETERS = 8_208_384
AUDIO_ENCODER_CONFIG_SHA256 = "d8c8d403350cf4e21b4351d3812fc61defc6c5784d2d8f8a622d4b8ca23b98e9"
AUDIO_PREPROCESSOR_SHA256 = "3e4dc53c73e7e3954fc4435034f5027733c61d7c123b38bbd1461598fd8503ef"
AUDIO_READOUT_SHA256 = "46808a882ab75f8f19a138e4a91d13ceb42cfdc82534c4608c42ee615774d3cb"
AUDIO_FEATURE_SIZE = 384
SPEECH_COMMANDS_CLASSES = (
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pcm_wav(path: Path) -> np.ndarray:
    """Read the frozen task's uncompressed mono PCM16 16 kHz input format."""
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16_000
            or stream.getcomptype() != "NONE"
        ):
            raise ValueError("expected uncompressed mono 16-bit 16 kHz PCM WAV")
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
    if samples.size == 0:
        raise ValueError("audio file is empty")
    return samples.astype(np.float32) / 32_768.0


class TextFreeAudioRuntime:
    """Map raw fixed-format audio to supplied typed command options.

    Cache keys are SHA-256 hashes of the source WAV. Cached values are immutable
    384-wide CPU tensors from this exact encoder/preprocessor instance and are
    reused across option subsets and permutations.
    """

    def __init__(
        self,
        encoder_dir: Path,
        readout_path: Path,
        *,
        device: str | torch.device = "cpu",
        cache_entries: int = 64,
        temperature: float = 1.0,
        expected_readout_sha256: str | None = None,
    ) -> None:
        if type(cache_entries) is not int or cache_entries < 0:
            raise ValueError("cache_entries must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if expected_readout_sha256 is None:
            expected_readout_sha256 = AUDIO_READOUT_SHA256
        if (
            not isinstance(expected_readout_sha256, str)
            or len(expected_readout_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_readout_sha256)
        ):
            raise ValueError("expected_readout_sha256 must be a lowercase SHA-256 hex digest")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("runtime supports CPU or CUDA")

        encoder_dir = encoder_dir.resolve(strict=True)
        config_path = encoder_dir / "config.json"
        weights_path = encoder_dir / "encoder.safetensors"
        processor_path = encoder_dir / "preprocessor_config.json"
        readout_path = readout_path.resolve(strict=True)
        for path in (config_path, weights_path, processor_path, readout_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        expected_hashes = {
            config_path: AUDIO_ENCODER_CONFIG_SHA256,
            weights_path: AUDIO_ENCODER_SHA256,
            processor_path: AUDIO_PREPROCESSOR_SHA256,
            readout_path: expected_readout_sha256,
        }
        for path, expected in expected_hashes.items():
            if sha256_file(path) != expected:
                raise ValueError(f"pinned audio artifact hash mismatch: {path.name}")

        config = WhisperConfig.from_pretrained(encoder_dir, local_files_only=True)
        self.encoder = WhisperEncoder(config).eval().to(self.device)
        state = load_file(str(weights_path), device="cpu")
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.requires_grad_(False)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            encoder_dir, local_files_only=True
        )
        if sha256_file(processor_path) != AUDIO_PREPROCESSOR_SHA256:
            raise ValueError("audio preprocessor changed during load")

        self.readout = TextFreeAudioReadout(AUDIO_FEATURE_SIZE, list(SPEECH_COMMANDS_CLASSES))
        readout_state = torch.load(readout_path, map_location="cpu", weights_only=True)
        if not isinstance(readout_state, dict):
            raise ValueError("audio readout checkpoint must be a state dictionary")
        self.readout.load_state_dict(readout_state, strict=True)
        self.readout.eval().to(self.device).requires_grad_(False)
        self.temperature = temperature
        self.audio_cache = BoundedFeatureCache(cache_entries) if cache_entries else None
        self.audio_encoder_calls = 0
        self.cache_metrics: dict[str, Any] = {}
        self.parameter_count = (
            sum(parameter.numel() for parameter in self.encoder.parameters())
            + sum(parameter.numel() for parameter in self.readout.parameters())
        )
        if sum(parameter.numel() for parameter in self.encoder.parameters()) != (
            AUDIO_ENCODER_PARAMETERS
        ):
            raise ValueError("unexpected Whisper encoder parameter count")
        if self.parameter_count != AUDIO_ENCODER_PARAMETERS + 3_850:
            raise ValueError("unexpected text-free audio runtime parameter count")
        if any("decoder" in name for name, _ in self.encoder.named_parameters()):
            raise ValueError("encoder-only runtime unexpectedly contains decoder parameters")

    def _encode_audio(self, path: Path) -> Tensor:
        waveform = read_pcm_wav(path)
        encoded = self.feature_extractor(
            waveform,
            sampling_rate=16_000,
            padding="max_length",
            max_length=self.feature_extractor.n_samples,
            truncation=True,
            return_tensors="pt",
        )
        input_features = encoded.input_features.to(self.device)
        # Match the BF16 autocast used when the frozen training feature cache
        # was produced. Keep frontend feature extraction outside autocast.
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16
        ):
            hidden = self.encoder(input_features).last_hidden_state
        valid_tokens = min(
            hidden.shape[1], max(1, math.ceil(len(waveform) / 320))
        )
        feature = hidden[0, :valid_tokens].float().mean(dim=0).cpu().contiguous()
        if feature.shape != (AUDIO_FEATURE_SIZE,) or not torch.isfinite(feature).all():
            raise ValueError("Whisper encoder produced invalid pooled audio features")
        return feature

    def encode_observation(
        self, audio_path: Path, *, expected_media_sha256: str | None = None
    ) -> tuple[Tensor, dict[str, Any]]:
        """Encode once per content hash and return feature plus auditable cache counters."""
        audio_path = audio_path.resolve(strict=True)
        digest = sha256_file(audio_path)
        if expected_media_sha256 is not None and digest != expected_media_sha256:
            raise ValueError("audio hash differs from the supplied frozen identity")
        cached = self.audio_cache.get(digest) if self.audio_cache is not None else None
        if cached is None:
            feature = self._encode_audio(audio_path)
            self.audio_encoder_calls += 1
            if sha256_file(audio_path) != digest:
                raise ValueError("audio content changed while encoding")
            feature_hash = hashlib.sha256(feature.numpy().tobytes()).hexdigest()
            if self.audio_cache is not None:
                self.audio_cache.put(digest, feature, feature_hash)
        else:
            feature, feature_hash = cached
        metrics = {
            "cache_hit": cached is not None,
            "media_sha256": digest,
            "feature_sha256": feature_hash,
            "cache_entries": len(self.audio_cache) if self.audio_cache is not None else 0,
            "single_event_encoder_calls": self.audio_encoder_calls,
        }
        self.cache_metrics = metrics
        return feature.clone(), dict(metrics)

    def decide(
        self,
        audio_path: Path,
        options: list[dict[str, str]],
        *,
        expected_media_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Return typed probabilities without question, transcript, or option text."""
        feature, metrics = self.encode_observation(
            audio_path, expected_media_sha256=expected_media_sha256
        )
        decision = self.readout.decide(
            feature.to(self.device), options, temperature=self.temperature
        )
        return {**decision, "encoder_metrics": metrics}

    def close(self) -> None:
        self.audio_cache = None
        self.encoder = None
        self.readout = None
