"""Spatially pooled frozen VideoMAE features for a bounded Path A probe."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

GRID_SIZE = 4
TOKEN_WIDTH = 384
FEATURE_WIDTH = GRID_SIZE * GRID_SIZE * TOKEN_WIDTH
HIDDEN_WIDTH = 128
TASK_EMBED_WIDTH = 16
TASK_NAMES = list(CLEVR4_TAXONOMIES)
TASK_TO_INDEX = {task: index for index, task in enumerate(TASK_NAMES)}


def pool_spatiotemporal_tokens(
    tokens: Tensor,
    *,
    temporal_tokens: int,
    patch_height: int,
    patch_width: int,
    grid_size: int = GRID_SIZE,
) -> Tensor:
    """Mean over time and retain a fixed spatial grid from flattened patch tokens."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, tokens, channels]")
    if min(temporal_tokens, patch_height, patch_width, grid_size) <= 0:
        raise ValueError("temporal and spatial dimensions must be positive")
    batch, token_count, channels = tokens.shape
    expected = temporal_tokens * patch_height * patch_width
    if token_count != expected:
        raise ValueError(f"expected {expected} tokens, received {token_count}")
    if not torch.is_floating_point(tokens) or not torch.isfinite(tokens).all():
        raise ValueError("tokens must be finite floating point values")
    spatial = tokens.reshape(batch, temporal_tokens, patch_height, patch_width, channels).mean(
        dim=1
    )
    spatial = F.adaptive_avg_pool2d(spatial.permute(0, 3, 1, 2), (grid_size, grid_size))
    return spatial.permute(0, 2, 3, 1).reshape(batch, grid_size * grid_size * channels)


def videomae_spatial_features(
    encoder: nn.Module, videos: Tensor, *, grid_size: int = GRID_SIZE
) -> Tensor:
    """Run the pinned ViT blocks and keep normalized 4x4 spatial patch summaries."""
    if videos.ndim != 5:
        raise ValueError("videos must have shape [batch, channels, time, height, width]")
    patch_embed = encoder.patch_embed
    patch_size = patch_embed.patch_size
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    else:
        patch_height, patch_width = patch_size
    image_height, image_width = videos.shape[-2:]
    if image_height % patch_height or image_width % patch_width:
        raise ValueError("video image dimensions must be divisible by the patch size")
    grid_height = image_height // patch_height
    grid_width = image_width // patch_width
    hidden = patch_embed(videos)
    if encoder.pos_embed is not None:
        hidden = hidden + encoder.pos_embed.expand(hidden.shape[0], -1, -1).type_as(hidden)
    hidden = encoder.pos_drop(hidden)
    for block in encoder.blocks:
        hidden = block(hidden)
    spatial = pool_spatiotemporal_tokens(
        hidden,
        temporal_tokens=hidden.shape[1] // (grid_height * grid_width),
        patch_height=grid_height,
        patch_width=grid_width,
        grid_size=grid_size,
    )
    spatial = spatial.reshape(spatial.shape[0], grid_size * grid_size, -1)
    norm = getattr(encoder, "fc_norm", None)
    if norm is None:
        norm = getattr(encoder, "norm", None)
    if norm is not None:
        spatial = norm(spatial)
    spatial = spatial.flatten(start_dim=1)
    if not torch.isfinite(spatial).all():
        raise FloatingPointError("non-finite spatial VideoMAE features")
    return spatial


class SpatialGridClevr4Scorer(nn.Module):
    """Fixed-taxonomy option scorer over frozen 4x4 VideoMAE spatial features."""

    def __init__(self) -> None:
        super().__init__()
        self.task_embedding = nn.Embedding(len(TASK_NAMES), TASK_EMBED_WIDTH)
        self.network = nn.Sequential(
            nn.Linear(FEATURE_WIDTH + TASK_EMBED_WIDTH, HIDDEN_WIDTH),
            nn.GELU(),
            nn.Linear(HIDDEN_WIDTH, max(len(values) for values in CLEVR4_TAXONOMIES.values())),
        )

    def forward(self, image_features: Tensor, task_ids: Tensor) -> Tensor:
        if image_features.ndim != 2 or image_features.shape[-1] != FEATURE_WIDTH:
            raise ValueError(f"image_features must have shape [batch, {FEATURE_WIDTH}]")
        if task_ids.ndim != 1 or task_ids.shape[0] != image_features.shape[0]:
            raise ValueError("task_ids must contain one task index per feature row")
        return self.network(torch.cat([image_features, self.task_embedding(task_ids)], dim=-1))
