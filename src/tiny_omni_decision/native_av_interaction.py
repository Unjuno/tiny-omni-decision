"""Opt-in AV residual mechanics, not a trained or calibrated fusion model.

The gates control this added path only. The original baseline still consumes all
available blocks. No corpus, source IDs, labels or media loaders belong here.
"""

from collections.abc import Mapping

import torch
from torch import nn

from tiny_omni_decision.native_candidate import WIDTHS, NativeCandidateReadout


class ConditionalAVReadout(nn.Module):
    """Compact parallel AV exchange with candidate-conditioned residual gates.

    Takes ownership of the supplied baseline module without altering its weights.
    Absent audio/video bypasses the added path exactly. Callers omit missing blocks;
    invalid blocks retain the baseline's strict rejection semantics.
    """

    def __init__(self, baseline=None, *, exchange=True, gated=True):
        super().__init__()
        if (type(exchange) is not bool or type(gated) is not bool
                or (baseline is not None and type(baseline) is not NativeCandidateReadout)):
            raise ValueError("native baseline and explicit boolean policies required")
        self.baseline = NativeCandidateReadout() if baseline is None else baseline
        self._exchange, self._gated = exchange, gated
        self.audio_latents = nn.Parameter(torch.randn(8, 192) * 0.02)
        self.video_latents = nn.Parameter(torch.randn(8, 192) * 0.02)
        self.pool = nn.MultiheadAttention(192, 4, dropout=0, batch_first=True)
        self.audio_from_video = nn.MultiheadAttention(192, 4, dropout=0, batch_first=True)
        self.video_from_audio = nn.MultiheadAttention(192, 4, dropout=0, batch_first=True)
        self.audio_norm, self.video_norm = nn.LayerNorm(192), nn.LayerNorm(192)
        self.candidate_attention = nn.MultiheadAttention(192, 4, dropout=0, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(960, 64), nn.GELU(), nn.Linear(64, 2), nn.Sigmoid())
        self.correction = nn.Sequential(nn.Linear(960, 192), nn.GELU(), nn.Linear(192, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        # Permit a caller-owned baseline already moved to a different device/dtype.
        self.to(device=self.baseline.modality.weight.device,
                dtype=self.baseline.modality.weight.dtype)

    def get_extra_state(self):
        return {"schema": 1, "exchange": self._exchange, "gated": self._gated}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("AV checkpoint policy differs")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Reject policy mismatch before any parameter can be overwritten, even
        # with strict=False. Old baseline state belongs in self.baseline explicitly.
        if state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("AV checkpoint policy missing or different")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def parameter_ledger(self):
        allocated = sum(p.numel() for p in self.parameters())
        inactive = []
        if not self._exchange:
            inactive += [self.audio_from_video, self.video_from_audio,
                         self.audio_norm, self.video_norm]
        if not self._gated:
            inactive += [self.gate]
        return {
            "allocated_readout_parameters": allocated,
            "active_joint_parameters": allocated - sum(
                p.numel() for module in inactive for p in module.parameters()
            ),
            "requires_grad_parameters": sum(p.numel() for p in self.parameters()
                                            if p.requires_grad),
            "reference_frozen_encoder_parameters": 195_365_529,
            "reference_total_parameters": 195_365_529 + allocated,
        }

    @staticmethod
    def _attend(module, query, memory):
        return module(query.unsqueeze(0), memory.unsqueeze(0), memory.unsqueeze(0),
                      need_weights=False)[0].squeeze(0)

    @staticmethod
    def _features(q, a, v):
        return torch.cat([q, a, v, a * v, (a - v).abs()], dim=-1)

    def _memory(self, blocks):
        b = self.baseline
        return b.memory_norm(torch.cat([
            b.projections[block.kind](block.values[block.valid])
            + b.position(block.positions[block.valid])
            + b.modality.weight[list(WIDTHS).index(block.kind)]
            for block in blocks
        ]))

    def forward(self, blocks, queries, *, diagnostics=False):
        # Run the unchanged validator and baseline path first. The repeated
        # projection for the residual is an explicit cost, not claimed free.
        scores = self.baseline(blocks, queries)
        audio = [b for b in blocks if b.kind in ("speech", "event_audio")]
        video = [b for b in blocks if b.kind == "video"]
        if not audio or not video:
            return (scores, {"active": False, "gates": None}) if diagnostics else scores
        a = self._attend(self.pool, self.audio_latents, self._memory(audio))
        v = self._attend(self.pool, self.video_latents, self._memory(video))
        if self._exchange:
            # Both updates use the same original pair, avoiding directional order bias.
            a, v = (self.audio_norm(a + self._attend(self.audio_from_video, a, v)),
                    self.video_norm(v + self._attend(self.video_from_audio, v, a)))
        q = self.baseline.query(queries)
        a = self._attend(self.candidate_attention, q, a)
        v = self._attend(self.candidate_attention, q, v)
        gates = self.gate(self._features(q, a, v)) if self._gated else q.new_ones((len(q), 2))
        a, v = gates[:, :1] * a, gates[:, 1:] * v
        delta = self.correction(self._features(q, a, v)).squeeze(-1)
        scores = scores + delta
        if not torch.isfinite(scores).all() or not torch.isfinite(gates).all():
            raise ValueError("nonfinite conditional AV output")
        return (scores, {"active": True, "gates": gates, "correction": delta}) if diagnostics \
            else scores

    def forward_batched(self, blocks, queries, *, diagnostics=False):
        """Score a padded native-token batch without per-example attention calls.

        ``blocks`` maps each kind to ``(values, positions, valid)`` with shapes
        ``[batch, tokens, width]``, ``[batch, tokens, 3]`` and ``[batch, tokens]``.
        """
        if not isinstance(blocks, Mapping) or not 1 <= len(blocks) <= 16:
            raise ValueError("1..16 batched native token blocks required")
        reference = self.baseline.modality.weight
        if (
            not isinstance(queries, torch.Tensor)
            or queries.ndim != 3
            or not 2 <= queries.shape[1] <= 40
            or queries.shape[2] != 384
            or queries.dtype != reference.dtype
            or queries.device != reference.device
            or not torch.isfinite(queries).all()
        ):
            raise ValueError("finite batch x 2..40 x 384 candidate queries required")

        batch_size = len(queries)
        total_tokens = 0
        memories, masks, checked = [], [], {}
        for kind, payload in blocks.items():
            if kind not in WIDTHS or not isinstance(payload, (tuple, list)) or len(payload) != 3:
                raise ValueError("known batched block kind and values/positions/valid required")
            values, positions, valid = payload
            if (
                not all(isinstance(value, torch.Tensor) for value in payload)
                or values.ndim != 3
                or values.shape[0] != batch_size
                or values.shape[1] < 1
                or values.shape[2] != WIDTHS[kind]
                or positions.shape != (*values.shape[:2], 3)
                or valid.shape != values.shape[:2]
                or values.dtype != reference.dtype
                or positions.dtype != reference.dtype
                or valid.dtype != torch.bool
                or any(value.device != reference.device for value in payload)
                or not torch.isfinite(values).all()
                or not torch.isfinite(positions).all()
            ):
                raise ValueError("finite aligned batched tokens/coordinates/bool mask required")
            total_tokens += values.shape[1]
            if total_tokens > 8192:
                raise ValueError("8192 token cap exceeded; no silent truncation")
            checked[kind] = (values, positions, valid)
            kind_index = list(WIDTHS).index(kind)
            memories.append(
                self.baseline.projections[kind](values)
                + self.baseline.position(positions)
                + self.baseline.modality.weight[kind_index]
            )
            masks.append(valid)

        memory = self.baseline.memory_norm(torch.cat(memories, dim=1))
        memory_valid = torch.cat(masks, dim=1)
        if not memory_valid.any(dim=1).all():
            raise ValueError("every example requires at least one valid native token")
        query = self.baseline.query(queries)
        attended, _ = self.baseline.attention(
            query,
            memory,
            memory,
            key_padding_mask=~memory_valid,
            need_weights=False,
        )
        scores = self.baseline.score(
            torch.cat([query, attended, query * attended, (query - attended).abs()], dim=-1)
        ).squeeze(-1)

        audio = [kind for kind in checked if kind in ("speech", "event_audio")]
        video = [kind for kind in checked if kind == "video"]
        active = torch.zeros(batch_size, dtype=torch.bool, device=reference.device)
        if audio and video:
            audio_valid = torch.cat([checked[kind][2] for kind in audio], dim=1).any(dim=1)
            video_valid = torch.cat([checked[kind][2] for kind in video], dim=1).any(dim=1)
            active = audio_valid & video_valid
        row_indices = active.nonzero(as_tuple=False).flatten()
        if not len(row_indices):
            return (scores, {"active": active, "gates": None}) if diagnostics else scores

        def modality_memory(kinds):
            projected, modality_masks = [], []
            for kind in kinds:
                values, positions, valid = checked[kind]
                selected_values = values.index_select(0, row_indices)
                selected_positions = positions.index_select(0, row_indices)
                selected_valid = valid.index_select(0, row_indices)
                kind_index = list(WIDTHS).index(kind)
                projected.append(
                    self.baseline.projections[kind](selected_values)
                    + self.baseline.position(selected_positions)
                    + self.baseline.modality.weight[kind_index]
                )
                modality_masks.append(selected_valid)
            features = self.baseline.memory_norm(torch.cat(projected, dim=1))
            return features, torch.cat(modality_masks, dim=1)

        audio_memory, audio_mask = modality_memory(audio)
        video_memory, video_mask = modality_memory(video)
        active_count = len(row_indices)
        audio_query = self.audio_latents.unsqueeze(0).expand(active_count, -1, -1)
        video_query = self.video_latents.unsqueeze(0).expand(active_count, -1, -1)

        def attend(module, query_values, key_values, key_mask=None):
            return module(
                query_values,
                key_values,
                key_values,
                key_padding_mask=None if key_mask is None else ~key_mask,
                need_weights=False,
            )[0]

        a = attend(self.pool, audio_query, audio_memory, audio_mask)
        v = attend(self.pool, video_query, video_memory, video_mask)
        if self._exchange:
            original_a, original_v = a, v
            a = self.audio_norm(a + attend(self.audio_from_video, original_a, original_v))
            v = self.video_norm(v + attend(self.video_from_audio, original_v, original_a))
        q = query.index_select(0, row_indices)
        a = attend(self.candidate_attention, q, a)
        v = attend(self.candidate_attention, q, v)
        gates = self.gate(self._features(q, a, v)) if self._gated \
            else q.new_ones((*q.shape[:2], 2))
        weighted_a, weighted_v = gates[..., :1] * a, gates[..., 1:] * v
        delta = self.correction(self._features(q, weighted_a, weighted_v)).squeeze(-1)
        scores = scores.index_add(0, row_indices, delta)
        if not torch.isfinite(scores).all() or not torch.isfinite(gates).all():
            raise ValueError("nonfinite conditional AV output")
        return (scores, {"active": active, "gates": gates, "correction": delta}) \
            if diagnostics else scores
