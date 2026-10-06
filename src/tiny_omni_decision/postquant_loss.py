"""FP32 option-only distillation, with valid-choice masking before normalization."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .postquant_policy import LossWeights


def option_distillation_loss(student_logits: Tensor,
     teacher_logits: Tensor,
     targets: Tensor,
     valid_options: Tensor,
     weights: LossWeights) -> dict[str,
     Tensor]:
    """Sum KL/Brier across choices, then average examples; CE is per-example mean.

    Inputs: logits [batch, padded_choices], int64 targets [batch], boolean mask.
    Teacher logits are detached; padded values never affect gradients or losses.
    Temperature is 1.0. This function does not alter sampling or optimizer state.
    """
    if not all(isinstance(v,
         Tensor) for v in (student_logits,
         teacher_logits,
         targets,
         valid_options)):
        raise TypeError('all numerical inputs must be tensors')
    if (student_logits.ndim != 2
        or student_logits.shape[0] == 0
        or (not 2 <= student_logits.shape[1] <= 62)
        or (teacher_logits.shape != student_logits.shape)
        or (valid_options.shape != student_logits.shape)
        or (targets.shape != (student_logits.shape[0],

        ))):
        raise ValueError('require nonempty matching [batch, 2..62 options] and [batch] targets')
    if (not student_logits.is_floating_point()
        or not teacher_logits.is_floating_point()
        or valid_options.dtype != torch.bool
        or (targets.dtype != torch.int64)):
        raise TypeError('require real floating logits, boolean mask and int64 targets')
    device = student_logits.device
    if device.type not in ('cpu',
         'cuda') or any(v.device != device for v in (teacher_logits,
         targets,
         valid_options)):
        raise ValueError('inputs must share one CPU or CUDA device')
    if bool((valid_options.sum(-1) < 2).any()):
        raise ValueError('each example requires at least two valid options')
    if bool(((targets < 0) | (targets >= student_logits.shape[1])).any()):
        raise ValueError('target outside option range')
    if not bool(valid_options.gather(1, targets[:, None]).all()):
        raise ValueError('target points to a padded option')
    if not isinstance(weights, LossWeights):
        raise TypeError('weights must be validated LossWeights')
    with torch.autocast(device_type=device.type, enabled=False):
        s, t = (student_logits.float(), teacher_logits.detach().float())
        if not bool(
            torch.isfinite(s[valid_options]).all() & torch.isfinite(t[valid_options]).all()
        ):
            raise ValueError('active logits must be finite and representable in FP32')
        logp = F.log_softmax(s.masked_fill(~valid_options, float('-inf')), dim=-1)
        logq = F.log_softmax(t.masked_fill(~valid_options, float('-inf')), dim=-1)
        if not bool(
            torch.isfinite(logp[valid_options]).all() & torch.isfinite(logq[valid_options]).all()
        ):
            raise ValueError('FP32 normalization overflow')
        # Remove padded -inf BEFORE subtracting logs: zero times NaN is still NaN.
        logp, logq = (logp.masked_fill(~valid_options, 0), logq.masked_fill(~valid_options, 0))
        p = logp.exp().masked_fill(~valid_options, 0)
        q = logq.exp().masked_fill(~valid_options, 0)
        kl = (q * (logq - logp)).sum(-1).mean()
        ce = -logp.gather(1, targets[:, None]).mean()
        one_hot = F.one_hot(targets, num_classes=s.shape[1]).float()
        brier = ((p - one_hot).square() * valid_options).sum(-1).mean()
        total = weights.option_kl * kl + weights.cross_entropy * ce + weights.brier * brier
    if not bool(torch.isfinite(total)):
        raise ValueError('nonfinite weighted option loss')
    return {'total': total, 'option_kl': kl, 'cross_entropy': ce, 'brier': brier}
