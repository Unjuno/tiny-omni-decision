"""Opt-in AV residual mechanics, not a trained or calibrated fusion model.

The gates control this added path only. The original baseline still consumes all
available blocks. No corpus, source IDs, labels or media loaders belong here.
"""

import torch
from torch import nn

from tiny_omni_decision.native_candidate import WIDTHS, NativeCandidateReadout


class ConditionalAVReadout(nn.Module):
    """Compact parallel AV exchange with candidate-conditioned residual gates.

    Takes ownership of the supplied baseline module without altering its weights.
    Absent audio/video bypasses the added path exactly. Callers omit missing blocks;
    invalid blocks retain the baseline's strict rejection semantics.
    """

    def __init__(self, baseline=None, *, exchange=True, gated=True,
                 candidate_output_norm=False):
        super().__init__()
        if (type(exchange) is not bool or type(gated) is not bool
                or type(candidate_output_norm) is not bool
                or (baseline is not None and type(baseline) is not NativeCandidateReadout)):
            raise ValueError("native baseline and explicit boolean policies required")
        self.baseline = NativeCandidateReadout() if baseline is None else baseline
        self._exchange, self._gated = exchange, gated
        self._candidate_output_norm = candidate_output_norm
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
        # Appended after the legacy modules so the original initialization stream
        # and checkpoint tensor names stay identical when this ablation is off.
        self.candidate_norm = nn.LayerNorm(192) if candidate_output_norm else None
        # Permit a caller-owned baseline already moved to a different device/dtype.
        self.to(device=self.baseline.modality.weight.device,
                dtype=self.baseline.modality.weight.dtype)

    def get_extra_state(self):
        if not self._candidate_output_norm:
            return {"schema": 1, "exchange": self._exchange, "gated": self._gated}
        return {"schema": 2, "exchange": self._exchange, "gated": self._gated,
                "candidate_output_norm": True}

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
        if not self._candidate_output_norm:
            inactive += [self.candidate_norm] if self.candidate_norm is not None else []
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
        baseline_scores = self.baseline(blocks, queries)
        audio = [b for b in blocks if b.kind in ("speech", "event_audio")]
        video = [b for b in blocks if b.kind == "video"]
        if not audio or not video:
            return (baseline_scores, {"active": False, "gates": None,
                                      "baseline_scores": baseline_scores,
                                      "query_only_scores": baseline_scores}) if diagnostics \
                else baseline_scores
        a = self._attend(self.pool, self.audio_latents, self._memory(audio))
        v = self._attend(self.pool, self.video_latents, self._memory(video))
        if self._exchange:
            # Both updates use the same original pair, avoiding directional order bias.
            a, v = (self.audio_norm(a + self._attend(self.audio_from_video, a, v)),
                    self.video_norm(v + self._attend(self.video_from_audio, v, a)))
        q = self.baseline.query(queries)
        a = self._attend(self.candidate_attention, q, a)
        v = self._attend(self.candidate_attention, q, v)
        if self._candidate_output_norm:
            a, v = self.candidate_norm(a), self.candidate_norm(v)
        gates = self.gate(self._features(q, a, v)) if self._gated else q.new_ones((len(q), 2))
        a, v = gates[:, :1] * a, gates[:, 1:] * v
        delta = self.correction(self._features(q, a, v)).squeeze(-1)
        scores = baseline_scores + delta
        if not torch.isfinite(scores).all() or not torch.isfinite(gates).all():
            raise ValueError("nonfinite conditional AV output")
        query_only = None
        if diagnostics:
            zero = torch.zeros_like(q)
            query_only = baseline_scores + self.correction(
                self._features(q, zero, zero)
            ).squeeze(-1)
        return (scores, {"active": True, "gates": gates, "correction": delta,
                         "candidate_output_norm": self._candidate_output_norm,
                         "baseline_scores": baseline_scores,
                         "query_only_scores": query_only}) if diagnostics \
            else scores
