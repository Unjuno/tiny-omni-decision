"""Reference ternary QAT and a single optional frozen-base LoRA.

PyTorch parametrizations preserve the original model graph. This is fake-quantized
floating-point *execution*, not a low-bit kernel. Packing is a separate operation.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


def _group_size(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("group_size must be a positive integer")


def quantize(weight: Tensor, group_size: int) -> tuple[Tensor, Tensor]:
    """Detached flattened absmean groups, round-to-nearest and clamp to three codes.

    Partial groups use their true element count, not the padded group length.
    Scales are FP32; all-zero groups use unit scale and all-zero codes.
    """
    _group_size(group_size)
    if not weight.is_floating_point() or weight.numel() == 0:
        raise ValueError("expected a nonempty floating-point weight")
    flat = weight.detach().float().reshape(-1)
    if not torch.isfinite(flat).all():
        raise ValueError("weights must be finite")
    count = flat.numel()
    groups = (count + group_size - 1) // group_size
    padded = torch.nn.functional.pad(flat, (0, groups * group_size - count))
    grouped = padded.reshape(groups, group_size)
    sizes = torch.full((groups,), group_size, dtype=torch.float32, device=flat.device)
    sizes[-1] = count - (groups - 1) * group_size
    scales = grouped.abs().sum(dim=1) / sizes
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    if not torch.isfinite(scales).all():
        raise ValueError("quantization scales must be finite")
    codes = (grouped / scales[:, None]).round().clamp(-1, 1).to(torch.int8)
    return codes.flatten()[:count].reshape(weight.shape), scales


def dequantize(codes: Tensor, scales: Tensor, group_size: int) -> Tensor:
    _group_size(group_size)
    count = codes.numel()
    if count == 0 or codes.dtype != torch.int8:
        raise ValueError("codes must be a nonempty int8 tensor")
    if scales.ndim != 1 or scales.numel() != (count + group_size - 1) // group_size:
        raise ValueError("invalid scale count")
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("scales must be finite and positive")
    if ((codes < -1) | (codes > 1)).any():
        raise ValueError("invalid ternary codes")
    return (codes.flatten().float() * scales.repeat_interleave(group_size)[:count]).reshape(
        codes.shape
    )


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, group_size: int) -> Tensor:
        codes, scales = quantize(weight, group_size)
        return dequantize(codes, scales, group_size).to(weight.dtype)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return gradient, None


class TernaryWeight(nn.Module):
    def __init__(self, group_size: int):
        super().__init__()
        _group_size(group_size)
        self.group_size = group_size
        self.register_buffer("codes", None)
        self.register_buffer("scales", None)

    @property
    def frozen(self) -> bool:
        return self.codes is not None

    def set_frozen(self, codes: Tensor, scales: Tensor) -> None:
        dequantize(codes, scales, self.group_size)  # validate before mutation
        self.codes = codes.detach().clone()
        self.scales = scales.detach().float().clone()

    def forward(self, weight: Tensor) -> Tensor:
        if self.frozen:
            return dequantize(self.codes, self.scales, self.group_size).to(weight.dtype)
        return _StraightThrough.apply(weight, self.group_size)


class LowRankDelta(nn.Module):
    """Reference low-rank weight correction; never merged into ternary codes."""
    def __init__(self, weight: Tensor, rank: int, alpha: float):
        super().__init__()
        if type(rank) is not int or rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive and finite")
        self.rank = min(rank, *weight.shape)
        self.alpha = float(alpha)
        self.lora_A = nn.Parameter(weight.new_empty((self.rank, weight.shape[1])))
        self.lora_B = nn.Parameter(weight.new_zeros((weight.shape[0], self.rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight: Tensor) -> Tensor:
        return weight + (self.lora_B @ self.lora_A) * (self.alpha / self.rank)


def owner_attribute(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    path, _, attribute = name.rpartition(".")
    return (model.get_submodule(path) if path else model), attribute


def _analyze_quantization_targets(
    model: nn.Module, group_size: int, exclude: Sequence[str] = ()
) -> tuple[dict[str, nn.Parameter], dict[str, nn.Parameter], dict[str, dict]]:
    _group_size(group_size)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    excluded = set(exclude)
    if excluded - parameters.keys():
        raise ValueError(f"unknown exclusion: {sorted(excluded - parameters.keys())}")
    if any(parametrize.is_parametrized(module) for module in model.modules()):
        raise ValueError("model is already parametrized")
    counts = Counter(id(parameter) for parameter in parameters.values())
    candidates = {
        name: parameter
        for name, parameter in parameters.items()
        if parameter.is_floating_point() and parameter.ndim >= 2 and name not in excluded
    }
    if not candidates:
        raise ValueError("no quantization targets")
    if any(counts[id(parameter)] != 1 for parameter in candidates.values()):
        raise ValueError(
            "shared target parameters require an explicit alias-aware implementation"
        )
    for name, weight in candidates.items():
        owner, _ = owner_attribute(model, name)
        if isinstance(owner, nn.Embedding) and owner.max_norm is not None:
            raise ValueError("Embedding max_norm would modify quantized weights in-place")
        if not weight.numel():
            raise ValueError(f"empty target: {name}")
        if not torch.isfinite(weight).all():
            raise ValueError(f"target {name} is not finite")
    exceptions = {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "bytes": parameter.numel() * parameter.element_size(),
            "reason": "explicit" if name in excluded else "non-matrix",
        }
        for name, parameter in parameters.items()
        if name not in candidates
    }
    return parameters, candidates, exceptions


def inspect_quantization_inventory(
    model: nn.Module, group_size: int = 128, exclude: Sequence[str] = ()
) -> dict:
    """Return the exact ternary target inventory without mutating the model graph."""
    _, candidates, exceptions = _analyze_quantization_targets(model, group_size, exclude)
    return {
        "group_size": group_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "targets": {name: list(parameter.shape) for name, parameter in candidates.items()},
        "exceptions": exceptions,
        "execution": "read_only_inventory_no_parametrization",
    }


class TernaryController:
    """Quantize all floating rank>=2 parameters except exact explicit exclusions.

    The inventory is built before graph mutation. Shared parameters are rejected
    rather than silently untied or double-counted. Non-target parameters are frozen.
    """
    def __init__(
        self, model: nn.Module, group_size: int = 128, exclude: Sequence[str] = ()
    ):
        _, candidates, exceptions = _analyze_quantization_targets(
            model, group_size, exclude
        )
        self.model = model
        self.group_size = group_size
        self.state_keys = tuple(model.state_dict().keys())
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.targets: dict[str, TernaryWeight] = {}
        self.loras: dict[str, LowRankDelta] = {}
        self.exceptions = exceptions
        model.requires_grad_(False)
        for name, weight in candidates.items():
            owner, attribute = owner_attribute(model, name)
            quantizer = TernaryWeight(group_size)
            weight.requires_grad_(True)
            parametrize.register_parametrization(owner, attribute, quantizer)
            self.targets[name] = quantizer

    def original(self, name: str) -> Tensor:
        owner, attribute = owner_attribute(self.model, name)
        return owner.parametrizations[attribute].original

    def components(self, name: str) -> tuple[Tensor, Tensor]:
        q = self.targets[name]
        if q.frozen:
            return q.codes.detach().clone(), q.scales.detach().clone()
        return quantize(self.original(name), self.group_size)

    def freeze(self) -> None:
        for name, q in self.targets.items():
            if not q.frozen:
                q.set_frozen(*self.components(name))
        self.model.requires_grad_(False)

    def add_lora(
        self, rank: int = 8, alpha: float = 16.0, targets: Sequence[str] | None = None
    ) -> None:
        if self.loras:
            raise ValueError("Recovery LoRA already attached")
        if not all(q.frozen for q in self.targets.values()):
            raise ValueError("freeze the quantized base before adding LoRA")
        if type(rank) is not int or rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("invalid LoRA rank or alpha")
        names = list(targets) if targets is not None else [
            n for n in self.targets if isinstance(owner_attribute(self.model, n)[0], nn.Linear)
        ]
        if not names or len(names) != len(set(names)):
            raise ValueError("LoRA targets must be nonempty and unique")
        for name in names:
            owner, attribute = owner_attribute(self.model, name)
            if (
                name not in self.targets or attribute != "weight"
                or not isinstance(owner, nn.Linear)
            ):
                raise ValueError(f"unsupported LoRA target: {name}")
        self.model.requires_grad_(False)
        for name in names:
            owner, attribute = owner_attribute(self.model, name)
            adapter = LowRankDelta(self.original(name), rank, alpha)
            parametrize.register_parametrization(owner, attribute, adapter)
            self.loras[name] = adapter

    def inventory(self) -> dict:
        return {
            "group_size": self.group_size,
            "parameter_count": self.parameter_count,
            "targets": {n: list(self.original(n).shape) for n in self.targets},
            "exceptions": self.exceptions,
            "execution": "pytorch_float_reference_not_packed_kernel",
        }
