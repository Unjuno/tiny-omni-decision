"""Exact shared fitted baseline, unchanged zero-initialized AV residual fixtures."""

import torch

from tiny_omni_decision.native_av_interaction import ConditionalAVReadout
from tiny_omni_decision.native_candidate import NativeCandidateReadout

ARMS = ('C0', 'E0G0', 'E0G1', 'E1G0', 'E1G1')


def initialized_model(arm, control_state, fixture_state):
    if arm not in ARMS:
        raise ValueError('registered baseline or factorial AV arm required')
    for state in (control_state, fixture_state):
        if not isinstance(state, dict) or not state:
            raise ValueError('complete CPU FP32 state required')
        for key, value in state.items():
            if key == '_extra_state':
                continue
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError('finite CPU FP32 initial tensors required')
    model = NativeCandidateReadout() if arm == 'C0' else ConditionalAVReadout(
        exchange=arm[1] == '1', gated=arm[3] == '1')
    model.load_state_dict(fixture_state, strict=True)
    baseline = model if arm == 'C0' else model.baseline
    baseline.load_state_dict(control_state, strict=True)
    if arm != 'C0' and (torch.count_nonzero(model.correction[-1].weight)
                       or torch.count_nonzero(model.correction[-1].bias)):
        raise ValueError('exact zero final correction required for initial control parity')
    count = sum(p.numel() for p in model.parameters())
    if count != (889345 if arm == 'C0' else 1732420) or 195365529+count > 200000000:
        raise ValueError('encoder-inclusive parameter budget differs')
    return model
