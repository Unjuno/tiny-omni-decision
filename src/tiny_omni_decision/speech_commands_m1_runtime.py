"""Task-scoped local runtime for the frozen fresh-speaker Speech Commands M1 readout."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .decision import DecisionOutput
from .speech_commands_m1_contract import (
    SPEECH_COMMANDS_M1_LABELS,
    SPEECH_COMMANDS_M1_READOUT_SHA256,
    SPEECH_COMMANDS_M1_TEMPERATURE,
)
from .textfree_audio import TextFreeAudioReadout

if TYPE_CHECKING:
    from .audio_runtime import SpeechCommandsMultiQueryRuntime


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SpeechCommandsM1Runtime:
    """Use the already-loaded frozen Whisper encoder and a hash-pinned M1 readout.

    This route only covers the ten fixed English Speech Commands labels. It
    requires the complete label set so caller-selected subsets cannot change
    the calibrated distribution.
    """

    def __init__(self, audio_runtime: SpeechCommandsMultiQueryRuntime, readout_path: Path) -> None:
        self.audio_runtime = audio_runtime
        self.readout_path = readout_path.resolve(strict=True)
        self.readout_sha256 = _sha256_file(self.readout_path)
        if self.readout_sha256 != SPEECH_COMMANDS_M1_READOUT_SHA256:
            raise ValueError("Speech Commands M1 readout hash mismatch")
        self.readout = TextFreeAudioReadout(384, list(SPEECH_COMMANDS_M1_LABELS)).to(
            audio_runtime.device
        )
        state = torch.load(self.readout_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Speech Commands M1 readout must be a state dictionary")
        self.readout.load_state_dict(state, strict=True)
        self.readout.eval()
        for parameter in self.readout.parameters():
            parameter.requires_grad_(False)
        self.temperature = SPEECH_COMMANDS_M1_TEMPERATURE
        self.parameter_count = sum(parameter.numel() for parameter in self.readout.parameters())
        if self.parameter_count != 3_850:
            raise ValueError(f"unexpected M1 readout parameter count: {self.parameter_count}")

    @torch.inference_mode()
    def decide(
        self, audio_path: Path, *, expected_media_sha256: str
    ) -> tuple[DecisionOutput, dict[str, object]]:
        if not isinstance(expected_media_sha256, str) or len(expected_media_sha256) != 64:
            raise ValueError("expected_media_sha256 must be a SHA-256 digest")
        before_digest = _sha256_file(audio_path)
        if before_digest.lower() != expected_media_sha256.lower():
            raise ValueError("audio hash differs from the supplied frozen identity")

        cache = self.audio_runtime.audio_cache
        cached = cache.get(before_digest) if cache is not None else None
        if cached is None:
            features = self.audio_runtime._encode_audio(audio_path)
            self.audio_runtime.audio_encoder_calls += 1
            if _sha256_file(audio_path) != before_digest:
                raise ValueError("audio content changed while encoding")
            feature_sha256 = hashlib.sha256(features.numpy().tobytes()).hexdigest()
            if cache is not None:
                cache.put(before_digest, features, feature_sha256)
        else:
            features, feature_sha256 = cached

        logits = self.readout(features.to(self.audio_runtime.device)).float()
        calibrated_logits = logits / self.temperature
        probabilities = torch.softmax(calibrated_logits, dim=-1).cpu()
        if (
            not torch.isfinite(calibrated_logits).all()
            or not torch.isfinite(probabilities).all()
            or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("Speech Commands M1 produced invalid option probabilities")
        predicted = int(probabilities.argmax().item())
        decision = DecisionOutput(
            option_logits=calibrated_logits.cpu(),
            probabilities=probabilities,
            predicted_index=predicted,
            confidence=float(probabilities[predicted]),
        )
        return decision, {
            "media_sha256": before_digest,
            "feature_sha256": feature_sha256,
            "encoder_calls": self.audio_runtime.audio_encoder_calls,
            "readout_sha256": self.readout_sha256,
            "temperature": self.temperature,
        }
