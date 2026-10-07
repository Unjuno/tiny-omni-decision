"""Symmetric groupwise ternary fake quantization and compact code packing."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

_TRITS_PER_BYTE = 5
_MAX_PACKED_TRIT_VALUE = 3**_TRITS_PER_BYTE
_TRIT_POWERS = (1, 3, 9, 27, 81)
_CUSTOM_AUDIO_CONV = "Gemma4AudioCausalConv1d"
_CUSTOM_TEXT_EMBEDDING = "EmbeddingGemma2TextScaledWordEmbedding"
_VISION_POSITION_PARAMETER = "vision_tower.patch_embedder.position_embedding_table"


def _validate_grouping(group_size: int, threshold_factor: float | None = None) -> None:
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    if threshold_factor is not None and (
        not math.isfinite(threshold_factor) or not 0.0 < threshold_factor < 1.0
    ):
        raise ValueError("threshold_factor must be finite and between zero and one")


def _validate_codes(codes: Tensor) -> None:
    if codes.dtype == torch.bool or codes.is_complex():
        raise ValueError("codes must contain only -1, 0, and +1")
    if codes.is_floating_point() and (
        not torch.isfinite(codes).all() or torch.any(codes != codes.round())
    ):
        raise ValueError("codes must contain only -1, 0, and +1")
    if torch.any((codes < -1) | (codes > 1)):
        raise ValueError("codes must contain only -1, 0, and +1")


def quantize_groupwise_ternary(
    weights: Tensor,
    *,
    group_size: int = 256,
    threshold_factor: float = 0.7,
) -> tuple[Tensor, Tensor]:
    """Return {-1, 0, +1} codes and active-value mean scales per flattened group."""
    _validate_grouping(group_size, threshold_factor)
    if not isinstance(weights, Tensor) or not weights.is_floating_point():
        raise ValueError("weights must be a floating-point tensor")
    if weights.numel() == 0:
        raise ValueError("weights must not be empty")

    values = weights.detach().float().reshape(-1)
    if not torch.isfinite(values).all():
        raise ValueError("weights must be finite")

    group_count = (values.numel() + group_size - 1) // group_size
    padding = group_count * group_size - values.numel()
    padded = nn.functional.pad(values, (0, padding))
    grouped = padded.reshape(group_count, group_size)
    valid = torch.arange(group_count * group_size, device=values.device).reshape(
        group_count, group_size
    ) < values.numel()
    absolute = grouped.abs()
    valid_counts = valid.sum(dim=1).clamp_min(1)
    means = (absolute * valid).sum(dim=1) / valid_counts
    thresholds = means * threshold_factor
    active = valid & (absolute > 0) & (absolute >= thresholds.unsqueeze(1))
    active_counts = active.sum(dim=1).clamp_min(1)
    scales = (absolute * active).sum(dim=1) / active_counts
    codes = torch.where(active, grouped.sign(), torch.zeros_like(grouped))
    return codes.reshape(-1)[: values.numel()].reshape(weights.shape).to(torch.int8), scales


def dequantize_groupwise_ternary(
    codes: Tensor,
    scales: Tensor,
    *,
    group_size: int = 256,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Restore scaled values from signed ternary codes and one scale per group."""
    _validate_grouping(group_size)
    if not isinstance(codes, Tensor) or codes.numel() == 0:
        raise ValueError("codes must be a non-empty tensor")
    _validate_codes(codes)
    expected_groups = (codes.numel() + group_size - 1) // group_size
    if not isinstance(scales, Tensor) or scales.ndim != 1 or scales.numel() != expected_groups:
        raise ValueError("scales must contain one value per code group")
    scale_values = scales.to(device=codes.device, dtype=torch.float32)
    if not torch.isfinite(scale_values).all() or torch.any(scale_values < 0):
        raise ValueError("scales must be finite and non-negative")
    group_ids = torch.arange(codes.numel(), device=codes.device) // group_size
    restored = codes.reshape(-1).float() * scale_values[group_ids]
    return restored.reshape(codes.shape).to(dtype=dtype or torch.float32)


def fake_quantize_ternary(
    weights: Tensor,
    *,
    group_size: int = 256,
    threshold_factor: float = 0.7,
) -> Tensor:
    """Use ternary values in the forward pass and an identity STE in backward."""
    codes, scales = quantize_groupwise_ternary(
        weights, group_size=group_size, threshold_factor=threshold_factor
    )
    quantized = dequantize_groupwise_ternary(
        codes, scales, group_size=group_size, dtype=weights.dtype
    )
    return weights + (quantized - weights).detach()


class _TernaryWeightParametrization(nn.Module):
    def __init__(self, *, group_size: int, threshold_factor: float) -> None:
        super().__init__()
        _validate_grouping(group_size, threshold_factor)
        self.group_size = group_size
        self.threshold_factor = threshold_factor

    def forward(self, weights: Tensor) -> Tensor:
        return fake_quantize_ternary(
            weights,
            group_size=self.group_size,
            threshold_factor=self.threshold_factor,
        )


def pack_ternary_codes(codes: Tensor) -> Tensor:
    """Pack five signed ternary codes per byte using little-endian base three."""
    if not isinstance(codes, Tensor) or codes.numel() == 0:
        raise ValueError("codes must be a non-empty tensor")
    _validate_codes(codes)
    values = codes.reshape(-1).to(torch.int16)
    digits = values + 1
    padding = (-digits.numel()) % _TRITS_PER_BYTE
    if padding:
        digits = nn.functional.pad(digits, (0, padding), value=1)
    powers = torch.tensor(_TRIT_POWERS, device=digits.device, dtype=torch.int16)
    packed = (digits.reshape(-1, _TRITS_PER_BYTE) * powers).sum(dim=1)
    return packed.to(torch.uint8)


def unpack_ternary_codes(packed: Tensor, *, num_elements: int) -> Tensor:
    """Decode packed base-three bytes and reject invalid values or padding."""
    if not isinstance(packed, Tensor) or packed.ndim != 1 or packed.dtype != torch.uint8:
        raise ValueError("packed codes must be a one-dimensional uint8 tensor")
    if not isinstance(num_elements, int) or isinstance(num_elements, bool) or num_elements < 1:
        raise ValueError("num_elements must be a positive integer")
    expected_bytes = (num_elements + _TRITS_PER_BYTE - 1) // _TRITS_PER_BYTE
    if packed.numel() != expected_bytes:
        raise ValueError("packed byte count does not match num_elements")
    values = packed.to(torch.int16)
    if torch.any(values >= _MAX_PACKED_TRIT_VALUE):
        raise ValueError("invalid packed trit byte: values must be less than 243")
    powers = torch.tensor(_TRIT_POWERS, device=packed.device, dtype=torch.int16)
    digits = torch.remainder(values.unsqueeze(1) // powers, 3)
    remainder = num_elements % _TRITS_PER_BYTE
    if remainder and torch.any(digits[-1, remainder:] != 1):
        raise ValueError("padding trits must encode zero")
    return (digits.reshape(-1)[:num_elements] - 1).to(torch.int8)


def select_ternary_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Select loaded matrix weights and fail closed on unknown matrix parameters."""
    modules = dict(model.named_modules())
    selected: list[str] = []
    unclassified: list[str] = []

    for name, parameter in model.named_parameters():
        parent_name, _, leaf_name = name.rpartition(".")
        module = modules[parent_name]
        class_name = type(module).__name__
        is_standard_weight = leaf_name == "weight" and parameter.ndim >= 2 and isinstance(
            module, (nn.Linear, nn.Conv2d, nn.Embedding)
        )
        is_custom_embedding_weight = (
            leaf_name == "weight"
            and parameter.ndim >= 2
            and class_name == _CUSTOM_TEXT_EMBEDDING
        )
        is_audio_conv_weight = (
            leaf_name == "weight"
            and parameter.ndim >= 2
            and class_name == _CUSTOM_AUDIO_CONV
        )
        is_vision_position_weight = (
            name == _VISION_POSITION_PARAMETER
            and leaf_name == "position_embedding_table"
            and parameter.ndim >= 2
        )
        if (
            is_standard_weight
            or is_custom_embedding_weight
            or is_audio_conv_weight
            or is_vision_position_weight
        ):
            selected.append(name)
        elif parameter.ndim >= 2:
            unclassified.append(name)

    if unclassified:
        raise ValueError(
            "unclassified matrix parameter(s): " + ", ".join(sorted(unclassified))
        )
    if not selected:
        raise ValueError("model has no supported ternary target parameters")
    return tuple(sorted(selected))


def apply_ternary_qat(
    model: nn.Module,
    *,
    group_size: int = 256,
    threshold_factor: float = 0.7,
) -> tuple[str, ...]:
    """Parametrize every verified target so each forward uses ternary weights."""
    _validate_grouping(group_size, threshold_factor)
    target_names = select_ternary_parameter_names(model)
    modules = dict(model.named_modules())
    for name in target_names:
        parent_name, _, leaf_name = name.rpartition(".")
        module = modules[parent_name]
        if parametrize.is_parametrized(module, leaf_name):
            raise ValueError(f"parameter is already parametrized: {name}")
        parametrize.register_parametrization(
            module,
            leaf_name,
            _TernaryWeightParametrization(
                group_size=group_size,
                threshold_factor=threshold_factor,
            ),
        )
    return target_names
