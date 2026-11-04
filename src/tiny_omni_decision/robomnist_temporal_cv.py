"""Compact independent-stream recurrent readout for ordered-match diagnostics."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def normalize_sequences(features: dict[str, tuple[np.ndarray, np.ndarray]], fit_groups: set[str]):
    if not fit_groups or not fit_groups <= features.keys():
        raise ValueError("fit groups must be a nonempty subset")
    arrays = {}
    means, scales = {}, {}
    for index, modality in enumerate(("audio", "video")):
        selected = []
        for group in sorted(fit_groups):
            x = np.asarray(features[group][index], dtype=np.float32)
            if x.ndim != 2 or x.shape[0] != 8 or not np.isfinite(x).all():
                raise ValueError("expected finite [8,width] feature sequences")
            selected.append(x)
        joined = np.concatenate(selected, axis=0)
        means[modality] = joined.mean(0)
        scales[modality] = joined.std(0)
        scales[modality][scales[modality] < 1e-6] = 1.0
    for group, streams in features.items():
        arrays[group] = tuple(
            ((np.asarray(x, dtype=np.float32) - means[name]) / scales[name]).astype(np.float32)
            for x, name in zip(streams, ("audio", "video"), strict=True)
        )
    normalizers = {
        f"{name}_{part}": v
        for name in means
        for part, v in (("mean", means[name]), ("scale", scales[name]))
    }
    return arrays, normalizers


def permute_by_group(features, *, seed=17):
    output = {}
    for group, streams in features.items():
        permutations = []
        for modality in ("audio", "video"):
            token = hashlib.sha256(f"{seed}:{modality}:{group}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
            permutations.append(rng.permutation(8))
        output[group] = tuple(x[p].copy() for x, p in zip(streams, permutations, strict=True))
    return output


class TemporalAVMatchScorer(nn.Module):
    """Separate-stream LSTMs; does not assert audio/video clock alignment."""

    def __init__(self, *, audio_width=384, video_width=768, hidden_size=16):
        super().__init__()
        self.audio_width, self.video_width = audio_width, video_width
        self.audio_lstm = nn.LSTM(audio_width, hidden_size, batch_first=True)
        self.video_lstm = nn.LSTM(video_width, hidden_size, batch_first=True)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, audio: Tensor, video: Tensor):
        if (
            audio.ndim != 3
            or video.ndim != 3
            or audio.shape[:2] != video.shape[:2]
            or audio.shape[1] != 8
            or audio.shape[2] != self.audio_width
            or video.shape[2] != self.video_width
        ):
            raise ValueError("expected audio/video sequences [batch,8,width]")
        if (
            audio.device != video.device
            or audio.dtype != video.dtype
            or not audio.is_floating_point()
            or not torch.isfinite(audio).all()
            or not torch.isfinite(video).all()
        ):
            raise ValueError("sequences must be finite matching float tensors")
        _, (audio_state, _) = self.audio_lstm(audio)
        _, (video_state, _) = self.video_lstm(video)
        a = F.normalize(audio_state[-1], p=2, dim=-1)
        v = F.normalize(video_state[-1], p=2, dim=-1)
        score = (a * v).sum(-1) * (32**0.5)
        return torch.stack((-score, score), dim=-1)
