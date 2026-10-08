"""Option-level supervision contracts for ternary student recovery."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .schema import DecisionExample


def build_ternary_qat_optimizer(
    parameters: list[torch.nn.Parameter],
    *,
    name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Build a QAT optimizer with an explicit optimizer-state memory policy."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not parameters:
        raise ValueError("QAT optimizer requires at least one trainable parameter")
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            fused=True,
        )
    if name == "adafactor":
        return torch.optim.Adafactor(
            parameters,
            lr=learning_rate,
            beta2_decay=-0.8,
            eps=(None, 1e-3),
            d=1.0,
            weight_decay=0.0,
            foreach=False,
        )
    raise ValueError(f"unsupported QAT optimizer: {name}")


def make_fp32_cpu_master_parameters(
    parameters: list[torch.nn.Parameter],
) -> list[torch.nn.Parameter]:
    """Create CPU FP32 master weights for lower-precision QAT shadow parameters."""
    if not parameters:
        raise ValueError("CPU master weights require at least one shadow parameter")
    return [
        torch.nn.Parameter(parameter.detach().to(device="cpu", dtype=torch.float32))
        for parameter in parameters
    ]


def copy_shadow_gradients_to_masters(
    shadows: list[torch.nn.Parameter],
    masters: list[torch.nn.Parameter],
    *,
    check_finite: bool = True,
) -> list[int]:
    """Transfer only present gradients to CPU FP32 masters and return active indices."""
    if len(shadows) != len(masters):
        raise ValueError("shadow and master parameter counts differ")
    active: list[int] = []
    for index, (shadow, master) in enumerate(zip(shadows, masters, strict=True)):
        if shadow.shape != master.shape or master.device.type != "cpu":
            raise ValueError("shadow/master device or shape contract is invalid")
        if shadow.grad is None:
            master.grad = None
            continue
        if check_finite and not torch.isfinite(shadow.grad).all():
            raise ValueError("shadow gradient must be finite")
        master.grad = shadow.grad.detach().to(device="cpu", dtype=torch.float32)
        active.append(index)
    return active


def sync_cpu_masters_to_shadows(
    shadows: list[torch.nn.Parameter],
    masters: list[torch.nn.Parameter],
    active_indices: list[int],
) -> None:
    """Copy updated FP32 master values back into the model's lower-precision shadows."""
    if len(shadows) != len(masters):
        raise ValueError("shadow and master parameter counts differ")
    if any(index < 0 or index >= len(shadows) for index in active_indices):
        raise ValueError("active master index is out of range")
    with torch.no_grad():
        for index in active_indices:
            shadow, master = shadows[index], masters[index]
            if shadow.shape != master.shape or master.device.type != "cpu":
                raise ValueError("shadow/master device or shape contract is invalid")
            shadow.copy_(master.to(device=shadow.device, dtype=shadow.dtype))


def restore_qat_shadows(
    model: torch.nn.Module,
    target_names: tuple[str, ...],
    weights: dict[str, Tensor],
) -> None:
    """Restore selected BF16 QAT shadow weights, validating all tensors first.

    Used only for an explicitly marked optimizer-reset warm start. This does
    not restore optimizer moments or constitute an exact training resume.
    """
    if not target_names or len(set(target_names)) != len(target_names):
        raise ValueError("warm-start targets must be nonempty and unique")
    if set(weights) != set(target_names):
        raise ValueError("warm-start shadow names differ from ternary target inventory")
    modules = dict(model.named_modules())
    copies: list[tuple[torch.nn.Parameter, Tensor]] = []
    for name in target_names:
        module_path, _, leaf = name.rpartition(".")
        module = modules.get(module_path)
        if module is None or not hasattr(module, "parametrizations"):
            raise ValueError(f"missing QAT shadow: {name}")
        chain = getattr(module.parametrizations, leaf, None)
        if chain is None:
            raise ValueError(f"missing QAT shadow: {name}")
        original = chain.original
        value = weights[name]
        if tuple(value.shape) != tuple(original.shape):
            raise ValueError(f"warm-start shadow shape mismatch: {name}")
        if value.dtype != original.dtype:
            raise ValueError(f"warm-start shadow dtype mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"warm-start shadow contains non-finite values: {name}")
        copies.append((original, value))
    with torch.no_grad():
        for original, value in copies:
            original.copy_(value.to(device=original.device))


def _option_order_sha256(options: list[str]) -> str:
    payload = json.dumps(options, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_teacher_option_cache(
    path: str | Path,
    training_examples: list[DecisionExample],
    *,
    expected_teacher_id: str,
    expected_teacher_revision: str,
    expected_temperature: float,
) -> dict[str, dict[str, Any]]:
    """Load a cache only when every record matches the exact train example and Teacher."""
    if not expected_teacher_id or not expected_teacher_revision:
        raise ValueError("expected Teacher identity is required")
    if not math.isfinite(expected_temperature) or expected_temperature <= 0:
        raise ValueError("expected Teacher temperature must be finite and positive")

    examples_by_id: dict[str, DecisionExample] = {}
    for example in training_examples:
        if example.split != "train":
            raise ValueError(f"teacher cache inputs must be train-only: {example.id}")
        if example.id in examples_by_id:
            raise ValueError(f"duplicate training sample id: {example.id}")
        examples_by_id[example.id] = example

    records: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid Teacher cache JSON at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Teacher cache row must be an object at line {line_number}")
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str):
                raise ValueError(f"Teacher cache sample_id must be a string at line {line_number}")
            if sample_id not in examples_by_id:
                raise ValueError(f"unexpected Teacher cache sample: {sample_id}")
            if sample_id in records:
                raise ValueError(f"duplicate teacher cache sample: {sample_id}")
            example = examples_by_id[sample_id]
            if (
                record.get("options") != example.options
                or record.get("target") != example.target
                or isinstance(record.get("target_index"), bool)
                or not isinstance(record.get("target_index"), int)
                or record.get("target_index") != example.options.index(example.target)
                or record.get("option_order_sha256") != _option_order_sha256(example.options)
            ):
                raise ValueError(f"option order mismatch for {sample_id}")
            if (
                record.get("teacher_id") != expected_teacher_id
                or record.get("teacher_revision") != expected_teacher_revision
            ):
                raise ValueError(f"Teacher identity mismatch for {sample_id}")

            temperature = record.get("teacher_temperature")
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature)
                or not math.isclose(temperature, expected_temperature, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(f"Teacher temperature mismatch for {sample_id}")
            option_count = len(example.options)
            logits = record.get("teacher_option_logits")
            probabilities = record.get("teacher_option_probabilities")
            if (
                not isinstance(logits, list)
                or not isinstance(probabilities, list)
                or len(logits) != option_count
                or len(probabilities) != option_count
            ):
                raise ValueError(f"Teacher option count mismatch for {sample_id}")
            if any(isinstance(value, bool) for value in [*logits, *probabilities]):
                raise ValueError(f"invalid Teacher option values for {sample_id}")
            try:
                logits = [float(value) for value in logits]
                probabilities = [float(value) for value in probabilities]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid Teacher option values for {sample_id}") from exc
            if (
                any(not math.isfinite(value) for value in logits)
                or any(
                    not math.isfinite(value) or value < 0.0 or value > 1.0
                    for value in probabilities
                )
                or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-4)
            ):
                raise ValueError(f"invalid Teacher option distribution for {sample_id}")

            scaled = [value / float(temperature) for value in logits]
            maximum = max(scaled)
            exponentials = [math.exp(value - maximum) for value in scaled]
            normalizer = sum(exponentials)
            expected_probabilities = [value / normalizer for value in exponentials]
            if any(
                abs(expected - actual) > 2e-3
                for expected, actual in zip(expected_probabilities, probabilities, strict=True)
            ):
                raise ValueError(f"Teacher logits/probabilities disagree for {sample_id}")
            normalized = dict(record)
            normalized["teacher_option_logits"] = logits
            normalized["teacher_option_probabilities"] = probabilities
            records[sample_id] = normalized

    missing = set(examples_by_id) - set(records)
    if missing:
        raise ValueError(f"missing training sample in Teacher cache: {min(missing)}")
    return records


def student_option_distillation_loss(
    student_logits: Tensor,
    teacher_probabilities: Tensor,
    *,
    target_index: int,
    option_kl_weight: float = 1.0,
    cross_entropy_weight: float = 0.2,
    brier_weight: float = 0.2,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine Teacher option KL with labeled CE and Brier auxiliary losses."""
    if student_logits.ndim != 1 or student_logits.numel() < 2:
        raise ValueError("student_logits must be a vector with at least two options")
    if teacher_probabilities.ndim != 1 or teacher_probabilities.shape != student_logits.shape:
        raise ValueError("teacher probabilities must match the student option count")
    if (
        not isinstance(target_index, int)
        or isinstance(target_index, bool)
        or not 0 <= target_index < student_logits.numel()
    ):
        raise ValueError("target_index must identify one supplied option")
    weights = (option_kl_weight, cross_entropy_weight, brier_weight)
    if any(not math.isfinite(weight) or weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("loss weights must be finite, non-negative, and not all zero")

    logits = student_logits.float()
    teacher = teacher_probabilities.to(device=logits.device, dtype=torch.float32)
    if not torch.isfinite(logits).all():
        raise ValueError("student option logits must be finite")
    if (
        not torch.isfinite(teacher).all()
        or torch.any(teacher < 0.0)
        or torch.any(teacher > 1.0)
        or not torch.isclose(
            teacher.sum(), torch.ones((), device=teacher.device), atol=1e-4, rtol=0.0
        )
    ):
        raise ValueError("teacher probabilities must be a finite distribution")

    log_probabilities = F.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    option_kl = F.kl_div(log_probabilities, teacher, reduction="sum")
    target = torch.tensor([target_index], device=logits.device)
    cross_entropy = F.cross_entropy(logits.unsqueeze(0), target)
    one_hot = F.one_hot(target, num_classes=logits.numel()).to(probabilities.dtype).squeeze(0)
    brier = ((probabilities - one_hot) ** 2).sum()
    total = (
        option_kl_weight * option_kl
        + cross_entropy_weight * cross_entropy
        + brier_weight * brier
    )
    if not torch.isfinite(total):
        raise ValueError("student distillation loss is non-finite")
    return total, {"option_kl": option_kl, "cross_entropy": cross_entropy, "brier": brier}
