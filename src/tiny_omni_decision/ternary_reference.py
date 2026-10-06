"""Tensor-level ternary reference; ordinary int8 codes, NOT a packed runtime.

No model loader, module-name guessing, in-place weight mutation or QAT is here.
Recipe is the predeclared baseline, not an asserted optimal quantizer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
RECIPE = 'row_group_meanabs_strict_ternary_v1'
DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

def _recipe(group_size: int, threshold_multiplier: float) -> None:
    if type(group_size) is not int or group_size < 1:
        raise ValueError('group_size must be a positive integer')
    if type(threshold_multiplier) not in (int,
         float) or not math.isfinite(threshold_multiplier) or threshold_multiplier < 0:
        raise ValueError('threshold_multiplier must be finite and nonnegative')

@dataclass(frozen=True)
class TernaryTensor:
    codes: Tensor
    scales: Tensor
    shape: tuple[int, int]
    group_size: int
    tail_lengths: tuple[int, ...]
    threshold_multiplier: float
    recipe_id: str = RECIPE

    def validate(self) -> None:
        _recipe(self.group_size, self.threshold_multiplier)
        if (self.recipe_id != RECIPE
            or type(self.shape) is not tuple
            or len(self.shape) != 2
            or any((type(n) is not int
            or n <= 0 for n in self.shape))):
            raise ValueError('invalid ternary recipe or shape')
        rows, cols = self.shape
        tails = tuple((min(self.group_size, cols - j) for j in range(0, cols, self.group_size)))
        if (self.tail_lengths != tails
            or any((type(n) is not int for n in self.tail_lengths))
            or (not isinstance(self.codes,

             Tensor)) or (not isinstance(self.scales,
             Tensor))):
            raise ValueError('invalid tail lengths or tensor payload')
        if (self.codes.dtype != torch.int8
            or tuple(self.codes.shape) != self.shape
            or self.scales.dtype != torch.float32
            or (self.scales.shape != (rows,

             len(tails)))
                 or (self.codes.device != self.scales.device)
                 or self.scales.requires_grad
                 or (self.codes.layout != torch.strided)
                 or (self.scales.layout != torch.strided)):
            raise ValueError('invalid codes/scales representation')
        if (not bool(((self.codes >= -1) & (self.codes <= 1)).all())
            or not bool(torch.isfinite(self.scales).all())
            or bool((self.scales < 0).any())):
            raise ValueError('codes must be ternary and scales finite/nonnegative')
        for j, n in enumerate(tails):
            has_value = (self.codes[:, j * self.group_size:j * self.group_size + n] != 0).any(-1)
            if not bool((has_value == (self.scales[:, j] > 0)).all()):
                raise ValueError('nonzero codes and positive scales must agree')

    def to_state_dict(self) -> dict[str, Any]:
        self.validate()
        return {'codes': self.codes.detach().clone(),
             'scales': self.scales.detach().clone(),
             'shape': list(self.shape),
             'group_size': self.group_size,
             'tail_lengths': list(self.tail_lengths),
             'threshold_multiplier': self.threshold_multiplier,
             'recipe_id': self.recipe_id}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> TernaryTensor:
        keys = {'codes',
             'scales',
             'shape',
             'group_size',
             'tail_lengths',
             'threshold_multiplier',
             'recipe_id'}
        if not isinstance(state, dict) or set(state) != keys:
            raise ValueError('unexpected ternary state keys')
        if type(state['shape']) is not list or type(state['tail_lengths']) is not list:
            raise ValueError('shape and tail_lengths must be lists in serialized state')
        result = cls(**state | {'shape': tuple(state['shape']),
             'tail_lengths': tuple(state['tail_lengths'])})
        result.validate()
        return cls(result.codes.clone(),
             result.scales.clone(),
             result.shape,
             result.group_size,
             result.tail_lengths,
             result.threshold_multiplier,
             result.recipe_id)

    def storage_report(self) -> dict[str, Any]:
        self.validate()
        return {'packed': False,
             'code_bytes': self.codes.numel() * self.codes.element_size(),
             'scale_bytes': self.scales.numel() * self.scales.element_size(),
             'zero_count': int((self.codes == 0).sum()),
             'weight_count': self.codes.numel(),
             'scope': 'tensor_payload_only_excludes_serialization_and_other_model_components'}

@torch.no_grad()
def quantize_reference(weight: Tensor,
     *,
     group_size: int=128,
     threshold_multiplier: float=0.7) -> TernaryTensor:
    """Assign in FP32, strictly above threshold; tails contain only actual weights."""
    _recipe(group_size, threshold_multiplier)
    if (not isinstance(weight,
         Tensor)
             or weight.ndim != 2
             or weight.numel() == 0
             or (weight.dtype not in DTYPES)
             or (weight.layout != torch.strided)
             or (weight.device.type not in ('cpu',

         'cuda'))):
        raise ValueError('require a nonempty dense real floating matrix on CPU or CUDA')
    w = weight.detach().float()
    if not bool(torch.isfinite(w).all()):
        raise ValueError('weight must be finite and representable in FP32')
    rows, cols = w.shape
    tails = tuple((min(group_size, cols - j) for j in range(0, cols, group_size)))
    codes = torch.zeros_like(w, dtype=torch.int8)
    scales = torch.zeros((rows, len(tails)), dtype=torch.float32, device=w.device)
    for j, n in enumerate(tails):
        part = w[:, j * group_size:j * group_size + n]
        absolute = part.abs()
        threshold = absolute.mean(-1, keepdim=True) * threshold_multiplier
        retained = absolute > threshold
        count = retained.sum(-1, keepdim=True)
        scale = absolute.masked_fill(~retained, 0).sum(-1, keepdim=True) / count.clamp_min(1)
        if not bool(torch.isfinite(threshold).all() & torch.isfinite(scale).all()):
            raise ValueError('FP32 quantizer arithmetic overflow')
        codes[:,
             j * group_size:j * group_size + n] = part.sign().to(torch.int8).masked_fill(~retained,
             0)
        scales[:, j] = scale[:, 0]
    result = TernaryTensor(codes,
         scales,
         (rows,
         cols),
         group_size,
         tails,
         float(threshold_multiplier))
    result.validate()
    return result

@torch.no_grad()
def dequantize_reference(value: TernaryTensor, *, dtype: torch.dtype) -> Tensor:
    """Produce a new constant dense matrix; not an optimized low-bit matmul."""
    value.validate()
    if dtype not in DTYPES:
        raise ValueError('dequantization requires a supported floating dtype')
    output = torch.empty(value.shape, dtype=dtype, device=value.codes.device)
    for j, n in enumerate(value.tail_lengths):
        start = j * value.group_size
        output[:,
             start:start + n] = value.codes[:,
             start:start + n].float() * value.scales[:,
             j:j + 1]
    if not bool(torch.isfinite(output).all()):
        raise ValueError('dequantized scale overflows requested dtype')
    return output
