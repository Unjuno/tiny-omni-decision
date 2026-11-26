"""Frozen AVATAR audio/video feature scorer runtime.

This is the inference counterpart to the fixed 384-D audio + 768-D video
concatenation readout. It consumes already encoded features; raw-media decoding
and the pinned pretrained encoders remain an explicit upstream responsibility.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tiny_omni_decision.decision_json import (
    decision_json_abstention_record,
    decision_json_record,
)

AUDIO_FEATURE_SIZE = 384
VIDEO_FEATURE_SIZE = 768
HIDDEN_SIZE = 64


class _PairwiseScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(AUDIO_FEATURE_SIZE + VIDEO_FEATURE_SIZE, HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, pairs: Tensor) -> Tensor:
        return self.layers(pairs).squeeze(-1)


class AvatarAVDecisionRuntime(nn.Module):
    """Hash-loadable, frozen scorer returning JSON probabilities over clip IDs."""

    def __init__(
        self,
        *,
        audio_mean: Sequence[float] | np.ndarray | Tensor,
        audio_std: Sequence[float] | np.ndarray | Tensor,
        video_mean: Sequence[float] | np.ndarray | Tensor,
        video_std: Sequence[float] | np.ndarray | Tensor,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.model = _PairwiseScorer()
        self.register_buffer("audio_mean", self._stat(audio_mean, AUDIO_FEATURE_SIZE, "audio mean"))
        self.register_buffer("audio_std", self._scale(audio_std, AUDIO_FEATURE_SIZE, "audio std"))
        self.register_buffer("video_mean", self._stat(video_mean, VIDEO_FEATURE_SIZE, "video mean"))
        self.register_buffer("video_std", self._scale(video_std, VIDEO_FEATURE_SIZE, "video std"))
        self.to(device)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _stat(values: Sequence[float] | np.ndarray | Tensor, width: int, name: str) -> Tensor:
        result = torch.as_tensor(values, dtype=torch.float32).detach().clone()
        if result.shape != (width,) or not torch.isfinite(result).all():
            raise ValueError(f"{name} must be a finite vector of width {width}")
        return result

    @classmethod
    def _scale(cls, values: Sequence[float] | np.ndarray | Tensor, width: int, name: str) -> Tensor:
        result = cls._stat(values, width, name)
        if (result <= 0).any():
            raise ValueError(f"{name} must contain only positive values")
        return result

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @staticmethod
    def sha256_file(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        normalization_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        expected_normalization_sha256: str,
        device: str | torch.device = "cpu",
    ) -> AvatarAVDecisionRuntime:
        checkpoint_path = Path(checkpoint_path)
        normalization_path = Path(normalization_path)
        for path, expected, name in (
            (checkpoint_path, expected_checkpoint_sha256, "checkpoint"),
            (normalization_path, expected_normalization_sha256, "normalization"),
        ):
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"expected {name} SHA-256 must be 64 hexadecimal characters")
            if any(character not in "0123456789abcdef" for character in expected.lower()):
                raise ValueError(f"expected {name} SHA-256 must be 64 hexadecimal characters")
            if cls.sha256_file(path) != expected.lower():
                raise ValueError(f"{name} SHA-256 mismatch")
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        required = {"audio_mean", "audio_std", "video_mean", "video_std"}
        if not isinstance(normalization, dict) or not required.issubset(normalization):
            raise ValueError("normalization is missing required train-derived statistics")
        runtime = cls(
            audio_mean=normalization["audio_mean"],
            audio_std=normalization["audio_std"],
            video_mean=normalization["video_mean"],
            video_std=normalization["video_std"],
            device=device,
        )
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
            raise ValueError("checkpoint must be a plain model state dictionary")
        runtime.model.load_state_dict(state, strict=True)
        runtime.eval()
        for parameter in runtime.parameters():
            parameter.requires_grad_(False)
        return runtime

    @staticmethod
    def _as_feature_tensor(
        values: Sequence[float] | np.ndarray | Tensor,
        shape: tuple[int, ...],
        name: str,
        device: torch.device,
    ) -> Tensor:
        result = torch.as_tensor(values, dtype=torch.float32, device=device).detach()
        if result.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(result.shape)}")
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} must contain only finite values")
        return result

    def decide(
        self,
        audio_features: Sequence[float] | np.ndarray | Tensor | None,
        video_features: Sequence[Sequence[float]] | np.ndarray | Tensor | None,
        option_ids: Sequence[str],
        option_texts: Sequence[str],
        *,
        request_id: str,
    ) -> dict[str, Any]:
        if audio_features is None or video_features is None:
            return decision_json_abstention_record(
                request_id=request_id,
                decision_type="choice",
                option_ids=option_ids,
                option_texts=option_texts,
                reason="required_audio_or_video_evidence_missing",
            )
        if len(option_ids) < 2 or len(option_ids) != len(option_texts):
            raise ValueError("at least two option IDs and matching option texts are required")
        device = self.audio_mean.device
        audio = self._as_feature_tensor(
            audio_features, (AUDIO_FEATURE_SIZE,), "audio features", device
        )
        videos = torch.as_tensor(video_features, dtype=torch.float32, device=device).detach()
        expected_shape = (len(option_ids), VIDEO_FEATURE_SIZE)
        if videos.shape != expected_shape:
            raise ValueError(
                f"video features must have shape {expected_shape}, got {tuple(videos.shape)}"
            )
        if not torch.isfinite(videos).all():
            raise ValueError("video features must contain only finite values")
        normalized_audio = (audio - self.audio_mean) / self.audio_std
        normalized_videos = (videos - self.video_mean) / self.video_std
        pairs = torch.cat(
            [normalized_audio.unsqueeze(0).expand(len(option_ids), -1), normalized_videos],
            dim=-1,
        )
        with torch.inference_mode():
            logits = self.model(pairs).float()
            if not torch.isfinite(logits).all():
                raise ValueError("AVATAR scorer produced non-finite option logits")
            probabilities = torch.softmax(logits, dim=0).cpu().tolist()
        return decision_json_record(
            request_id=request_id,
            decision_type="choice",
            option_ids=option_ids,
            option_texts=option_texts,
            probabilities=probabilities,
        )
