"""Untrained frozen-feature Path A connection, not a qualified Omni model.

Encoders belong to the caller. This module has no downloads, model loads,
class vocabulary, calibration, implicit pooling or hidden decoder.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

WIDTHS = {"text": 384, "image": 768, "video": 768, "speech": 384, "event_audio": 384}


@dataclass(frozen=True)
class NativeTokenBlock:
    kind: str
    values: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor


class NativeCandidateReadout(nn.Module):
    """Candidate-equivariant cross attention over explicit native token blocks."""

    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleDict({k: nn.Linear(d, 192) for k, d in WIDTHS.items()})
        self.modality = nn.Embedding(len(WIDTHS), 192)
        self.position = nn.Linear(3, 192, bias=False)
        self.memory_norm = nn.LayerNorm(192)
        self.query = nn.Sequential(nn.Linear(384, 192), nn.LayerNorm(192))
        self.attention = nn.MultiheadAttention(192, 4, dropout=0, batch_first=True)
        self.score = nn.Sequential(nn.Linear(4 * 192, 192), nn.GELU(), nn.Linear(192, 1))

    def forward(self, blocks, queries):
        reference = self.modality.weight
        if (
            not isinstance(queries, torch.Tensor) or queries.ndim != 2
            or not 2 <= queries.shape[0] <= 40 or queries.shape[1] != 384
            or queries.dtype != reference.dtype or queries.device != reference.device
            or not torch.isfinite(queries).all()
        ):
            raise ValueError("finite2..40x384 queries on the model dtype/device required")
        if not isinstance(blocks, (list, tuple)) or not 1 <= len(blocks) <= 16:
            raise ValueError("1..16 native token blocks required")
        count = 0
        memories = []
        for block in blocks:
            if not isinstance(block, NativeTokenBlock) or block.kind not in WIDTHS:
                raise ValueError("known native token block required")
            x, pos, mask = block.values, block.positions, block.valid
            if (
                not all(isinstance(v, torch.Tensor) for v in (x, pos, mask))
                or x.ndim != 2 or x.shape[1] != WIDTHS[block.kind] or len(x) < 1
                or pos.shape != (len(x), 3) or mask.shape != (len(x),)
                or x.dtype != reference.dtype or pos.dtype != reference.dtype
                or mask.dtype != torch.bool
                or any(v.device != reference.device for v in (x, pos, mask))
                or not torch.isfinite(x).all() or not torch.isfinite(pos).all()
                or not mask.any()
            ):
                raise ValueError("finite aligned nonempty tokens/coordinates/bool mask required")
            count += len(x)
            if count > 8192:
                raise ValueError("8192 token cap exceeded; no silent truncation")
            kind = list(WIDTHS).index(block.kind)
            memories.append(self.projections[block.kind](x[mask])
                            + self.position(pos[mask]) + self.modality.weight[kind])
        memory = self.memory_norm(torch.cat(memories)).unsqueeze(0)
        q = self.query(queries)
        attended, _ = self.attention(q.unsqueeze(0), memory, memory, need_weights=False)
        attended = attended.squeeze(0)
        logits = self.score(torch.cat([q, attended, q * attended, (q - attended).abs()], -1))
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite native candidate scores")
        return logits.squeeze(-1)


def candidate_json(scores, option_ids):
    """T1 candidate probabilities; IDs only associate outputs, never affect scores."""
    if (
        not isinstance(scores, torch.Tensor) or scores.ndim != 1
        or not 2 <= len(scores) <= 40 or len(option_ids) != len(scores)
        or any(not isinstance(i, str) or not i.strip() for i in option_ids)
        or len(set(option_ids)) != len(option_ids) or not torch.isfinite(scores).all()
    ):
        raise ValueError("finite scores and2..40 unique nonempty IDs required")
    values = scores.detach().double().cpu().tolist()
    maximum = max(values)
    weights = [math.exp(v - maximum) for v in values]
    total = math.fsum(weights)
    probabilities = {i: v / total for i, v in zip(option_ids, weights, strict=True)}
    selected = min(i for i, v in zip(option_ids, values, strict=True) if v == maximum)
    return dict(probabilities=probabilities, selected_option_id=selected)
