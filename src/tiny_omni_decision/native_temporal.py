"""Optional feature-time memory; not a causal encoder or a qualified task model."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .native_candidate import WIDTHS, NativeCandidateReadout, NativeTokenBlock


class TemporalResidual(nn.Module):
    """Explicit-state LSTM. Invalid steps are omitted, including internal holes.

    The zero output projection preserves the baseline at initialization. Recurrent
    parameters start receiving gradients after that projection first changes.
    Returned state is caller-owned: never reused implicitly across examples.
    """

    def __init__(self, width, hidden=192):
        super().__init__()
        if type(width) is not int or type(hidden) is not int or min(width, hidden) < 1:
            raise ValueError("positive integer dimensions required")
        self.width, self.hidden = width, hidden
        self.input = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden))
        self.recurrent = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, values, valid, state=None):
        reference = self.output.weight
        if (not isinstance(values, torch.Tensor) or values.ndim != 3
                or min(values.shape[:2]) < 1 or values.shape[2] != self.width
                or not isinstance(valid, torch.Tensor) or valid.shape != values.shape[:2]
                or values.dtype != reference.dtype or valid.dtype != torch.bool
                or values.device != reference.device or valid.device != reference.device
                or not torch.isfinite(values).all()):
            raise ValueError("finite BxTxD values and bool validity on model dtype/device required")
        batch, steps = values.shape[:2]
        if state is None:
            state = tuple(values.new_zeros(1, batch, self.hidden) for _ in range(2))
        if (not isinstance(state, tuple) or len(state) != 2
                or any(not isinstance(s, torch.Tensor) or s.shape != (1, batch, self.hidden)
                       or s.dtype != values.dtype or s.device != values.device
                       or not torch.isfinite(s).all() for s in state)):
            raise ValueError("two finite explicit 1xBxH states on model dtype/device required")
        if valid.all():
            hidden, final = self.recurrent(self.input(values), state)
            result = values + self.output(hidden)
        else:
            active = valid.any(1).nonzero().flatten()
            if not len(active):
                return values, state
            # Projection/norm never sees masked values. Compress holes, then restore
            # the original coordinates. Empty streams keep their caller-owned state.
            sequences = [values[i, valid[i]] for i in active]
            lengths = torch.tensor([len(x) for x in sequences], dtype=torch.int64)
            packed = pack_padded_sequence(self.input(pad_sequence(sequences, batch_first=True)),
                                          lengths, batch_first=True, enforce_sorted=False)
            hidden, last = self.recurrent(packed, tuple(s[:, active] for s in state))
            hidden, _ = pad_packed_sequence(hidden, batch_first=True)
            locations = [i * steps + valid[i].nonzero().flatten() for i in active]
            updates = [self.output(hidden[j, :length]) for j, length in enumerate(lengths)]
            delta = values.new_zeros(batch * steps, self.width).index_copy(
                0, torch.cat(locations), torch.cat(updates))
            result = values + delta.view_as(values)
            final = tuple(s.index_copy(1, active, v) for s, v in zip(state, last, strict=True))
        if not torch.isfinite(result).all() or any(not torch.isfinite(s).all() for s in final):
            raise ValueError("nonfinite temporal output/state")
        return result, final


def temporal_layout(block):
    """Indices [independent spatial cells, ordered time], never spatial-as-time."""
    if not isinstance(block, NativeTokenBlock) or block.kind not in {"speech", "video"}:
        raise ValueError("speech/video block required")
    x, p, valid = block.values, block.positions, block.valid
    if (not all(isinstance(v, torch.Tensor) for v in (x, p, valid))
            or x.ndim != 2 or not 1 <= len(x) <= 8192 or x.shape[1] != WIDTHS[block.kind]
            or p.shape != (len(x), 3) or valid.shape != (len(x),)
            or not x.is_floating_point() or p.dtype != x.dtype or valid.dtype != torch.bool
            or p.device != x.device or valid.device != x.device
            or not torch.isfinite(x).all() or not torch.isfinite(p).all() or not valid.any()):
        raise ValueError("finite aligned native temporal block required")
    if block.kind == "speech":
        if not (p[1:, 0] > p[:-1, 0]).all() or (p[:, 1:] != 0).any():
            raise ValueError("strict speech time order and zero spatial coordinates required")
        return torch.arange(len(x), device=x.device).view(1, -1)
    times, counts = torch.unique_consecutive(p[:, 0], return_counts=True)
    if not (times[1:] > times[:-1]).all() or not (counts == counts[0]).all():
        raise ValueError("time-major complete repeated spatial layout required")
    space = int(counts[0])
    spatial = p[:, 1:].reshape(len(times), space, 2)
    if (len(torch.unique(spatial[0], dim=0)) != space
            or not torch.equal(spatial, spatial[:1].expand_as(spatial))):
        raise ValueError("unique stable spatial identities across all frames required")
    return torch.arange(len(x), device=x.device).reshape(len(times), space).T


class TemporalNativeCandidate(nn.Module):
    """Separate optional adapter + unchanged readout; reset each input block.

    Frozen full-clip features may contain future information. This wrapper makes
    no raw-stream causality, clock synchronization, or object-tracking claim.
    """

    def __init__(self, readout=None):
        super().__init__()
        self.readout = NativeCandidateReadout() if readout is None else readout
        if not isinstance(self.readout, NativeCandidateReadout):
            raise ValueError("native candidate readout required")
        self.temporal = nn.ModuleDict({k: TemporalResidual(WIDTHS[k])
                                       for k in ("video", "speech")})

    def encode(self, blocks):
        if (not isinstance(blocks, (list, tuple)) or not 1 <= len(blocks) <= 16
                or any(not isinstance(b, NativeTokenBlock)
                       or not isinstance(b.values, torch.Tensor) or b.values.ndim != 2
                       for b in blocks)
                or sum(len(b.values) for b in blocks) > 8192):
            raise ValueError("1..16 native blocks and <=8192 tokens required")
        result = []
        for block in blocks:
            if block.kind not in self.temporal:
                result.append(block)
                continue
            indices = temporal_layout(block)
            enhanced, _ = self.temporal[block.kind](block.values[indices], block.valid[indices])
            restored = torch.empty_like(block.values).index_copy(
                0, indices.flatten(), enhanced.reshape(-1, block.values.shape[1]))
            result.append(NativeTokenBlock(block.kind, restored, block.positions, block.valid))
        return result

    def forward(self, blocks, queries):
        return self.readout(self.encode(blocks), queries)
