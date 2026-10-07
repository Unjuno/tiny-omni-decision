"""Symmetric groupwise ternary fake quantization and compact code packing."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

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
    return quantized.detach() + (weights - weights.detach())


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


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_packed_ternary_overlay(
    model: nn.Module,
    output_dir: str | Path,
    *,
    base_model_id: str,
    base_revision: str,
    group_size: int = 256,
    threshold_factor: float = 0.7,
) -> dict[str, object]:
    """Write packed ternary weights as an immutable overlay for a pinned base model.

    The overlay contains only packed target codes and FP32 group scales. The base
    checkpoint remains an external dependency; exceptions are inventoried in the
    manifest and retain their original high-precision base values.
    """
    _validate_grouping(group_size, threshold_factor)
    if not base_model_id.strip() or not base_revision.strip():
        raise ValueError("base_model_id and base_revision are required")

    targets = select_ternary_parameter_names(model)
    parameters = dict(model.named_parameters())
    modules = dict(model.named_modules())
    for name in targets:
        parent_name, _, leaf_name = name.rpartition(".")
        if parametrize.is_parametrized(modules[parent_name], leaf_name):
            raise ValueError(f"cannot export a model with active QAT parametrization: {name}")

    tensor_file = Path(output_dir) / "weights.safetensors"
    manifest_file = Path(output_dir) / "manifest.json"
    if tensor_file.exists() or manifest_file.exists():
        raise FileExistsError(f"ternary overlay already exists: {output_dir}")
    tensor_file.parent.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file

    tensors: dict[str, Tensor] = {}
    records: list[dict[str, object]] = []
    for name in targets:
        parameter = parameters[name]
        codes, scales = quantize_groupwise_ternary(
            parameter, group_size=group_size, threshold_factor=threshold_factor
        )
        packed = pack_ternary_codes(codes).detach().cpu().contiguous()
        scale_tensor = scales.detach().to(device="cpu", dtype=torch.float32).contiguous()
        tensors[f"code__{name}"] = packed
        tensors[f"scale__{name}"] = scale_tensor
        records.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "elements": parameter.numel(),
                "packed_code_bytes": packed.numel(),
                "scale_bytes": _tensor_bytes(scale_tensor),
            }
        )

    temporary_file = tensor_file.with_name("weights.safetensors.tmp")
    save_file(tensors, str(temporary_file))
    os.replace(temporary_file, tensor_file)
    high_precision_parameters = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "bytes": _tensor_bytes(parameter),
            "reason": "non_matrix_parameter_or_unsupported_weight_role",
        }
        for name, parameter in sorted(parameters.items())
        if name not in targets
    ]
    high_precision_buffers = [
        {
            "name": name,
            "shape": list(buffer.shape),
            "dtype": str(buffer.dtype),
            "bytes": _tensor_bytes(buffer),
            "reason": "runtime_buffer_retained_from_pinned_base",
        }
        for name, buffer in sorted(model.named_buffers())
    ]
    payload_bytes = tensor_file.stat().st_size
    metadata: dict[str, object] = {
        "format": "tiny-omni-ternary-overlay-v1",
        "base_model_id": base_model_id,
        "base_revision": base_revision,
        "group_size": group_size,
        "threshold_factor": threshold_factor,
        "code_encoding": "five_signed_trits_per_byte_little_endian_base3",
        "scale_dtype": "torch.float32",
        "tensor_file": tensor_file.name,
        "tensor_file_sha256": _file_sha256(tensor_file),
        "tensor_file_bytes": payload_bytes,
        "target_parameter_count": len(records),
        "target_element_count": sum(int(record["elements"]) for record in records),
        "packed_code_bytes": sum(int(record["packed_code_bytes"]) for record in records),
        "scale_bytes": sum(int(record["scale_bytes"]) for record in records),
        "higher_precision_parameter_bytes": sum(
            int(record["bytes"]) for record in high_precision_parameters
        ),
        "runtime_buffer_bytes": sum(int(record["bytes"]) for record in high_precision_buffers),
        "targets": records,
        "higher_precision_parameters": high_precision_parameters,
        "runtime_buffers": high_precision_buffers,
        "base_checkpoint_is_external": True,
    }
    manifest_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_packed_ternary_overlay(
    model: nn.Module,
    overlay_dir: str | Path,
    *,
    expected_base_model_id: str,
    expected_base_revision: str,
) -> dict[str, object]:
    """Validate and apply an overlay to the matching unparametrized base model."""
    directory = Path(overlay_dir)
    manifest_path = directory / "manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "tiny-omni-ternary-overlay-v1":
        raise ValueError("unsupported ternary overlay format")
    if metadata.get("base_model_id") != expected_base_model_id:
        raise ValueError("base model id mismatch")
    if metadata.get("base_revision") != expected_base_revision:
        raise ValueError("base revision mismatch")

    tensor_file = directory / str(metadata["tensor_file"])
    if _file_sha256(tensor_file) != metadata.get("tensor_file_sha256"):
        raise ValueError("ternary overlay tensor hash mismatch")
    target_names = select_ternary_parameter_names(model)
    records = metadata.get("targets")
    if not isinstance(records, list):
        raise ValueError("ternary overlay target inventory is invalid")
    record_by_name = {str(record.get("name")): record for record in records}
    if len(record_by_name) != len(records) or set(record_by_name) != set(target_names):
        raise ValueError("ternary overlay targets do not match the loaded model")

    from safetensors.torch import load_file

    tensors = load_file(str(tensor_file), device="cpu")
    expected_keys = {
        key
        for name in target_names
        for key in (f"code__{name}", f"scale__{name}")
    }
    if set(tensors) != expected_keys:
        raise ValueError("ternary overlay tensor inventory does not match its manifest")

    parameters = dict(model.named_parameters())
    group_size = int(metadata["group_size"])
    with torch.no_grad():
        for name in target_names:
            parameter = parameters[name]
            record = record_by_name[name]
            if list(parameter.shape) != record.get("shape") or str(parameter.dtype) != record.get(
                "dtype"
            ):
                raise ValueError(f"base tensor shape or dtype mismatch: {name}")
            codes = unpack_ternary_codes(
                tensors[f"code__{name}"], num_elements=parameter.numel()
            ).reshape(parameter.shape)
            scales = tensors[f"scale__{name}"]
            restored = dequantize_groupwise_ternary(
                codes, scales, group_size=group_size, dtype=parameter.dtype
            )
            if list(restored.shape) != list(parameter.shape):
                raise ValueError(f"decoded tensor shape mismatch: {name}")
            parameter.copy_(restored.to(device=parameter.device))
    return metadata
