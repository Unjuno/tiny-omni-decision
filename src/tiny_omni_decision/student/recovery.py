"""Bounded QAT distillation, then validation-gated frozen-base LoRA.

Final held-out data are intentionally not accepted by the training entry point.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .artifacts import load_bundle, save_bundle
from .data import Cache, Record, assert_disjoint, split_identities, write_json
from .ternary import TernaryController


def distillation_loss(
    student: Tensor, teacher: Tensor, target: int, *, temperature: float = 1.0,
    ce_weight: float = 1.0, brier_weight: float = 0.2,
) -> Tensor:
    if student.ndim != 1 or teacher.shape != student.shape or student.numel() < 2:
        raise ValueError("one logit per corresponding option is required")
    if type(target) is not int or not 0 <= target < student.numel():
        raise ValueError("invalid target")
    if not math.isfinite(temperature) or temperature <= 0 or not all(
        math.isfinite(w) and w >= 0 for w in (ce_weight, brier_weight)
    ):
        raise ValueError("invalid loss temperature/weights")
    s = student.float()
    t = teacher.detach().to(s.device, dtype=torch.float32)
    if not torch.isfinite(s).all() or not torch.isfinite(t).all():
        raise ValueError("non-finite option logits")
    kd = F.kl_div(F.log_softmax(s / temperature, dim=-1),
                  F.softmax(t / temperature, dim=-1), reduction="sum") * temperature**2
    label = torch.tensor([target], device=s.device)
    ce = F.cross_entropy(s[None], label)
    target_probs = F.one_hot(label[0], s.numel()).float()
    brier = (F.softmax(s, dim=-1) - target_probs).square().sum()
    return kd + ce_weight * ce + brier_weight * brier


def evaluate(model: nn.Module | None, records: list[Record]) -> tuple[dict, list[list[float]]]:
    if not records:
        raise ValueError("empty evaluation")
    if model is not None:
        model.eval()
    groups = defaultdict(list)
    logits_list = []
    with torch.no_grad():
        for record in records:
            logits = torch.tensor(record.teacher_logits) if model is None else model(record)
            logits = logits.detach().float().cpu()
            if logits.shape != (len(record.options),) or not torch.isfinite(logits).all():
                raise ValueError(f"invalid option logits: {record.sample_id}")
            logits_list.append(logits.tolist())
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            predicted = int(p.argmax())
            target = F.one_hot(torch.tensor(record.target), len(record.options)).float()
            item = (float(predicted == record.target), float(-logp[record.target]),
                    float((p - target).square().sum()), float(p.max()))
            groups["all"].append(item)
            groups[record.modality].append(item)
    metrics = {}
    for name, items in groups.items():
        count = len(items)
        bins = defaultdict(list)
        for item in items:
            bins[min(int(item[3] * 15), 14)].append(item)
        ece = sum(abs(sum(i[0] - i[3] for i in b)) for b in bins.values()) / count
        metrics[name] = {"count": count, "accuracy": sum(i[0] for i in items) / count,
                         "nll": sum(i[1] for i in items) / count,
                         "brier": sum(i[2] for i in items) / count, "ece": ece}
    return metrics, logits_list


def quality_gate(student: dict, teacher: dict, limits: dict) -> dict:
    required = {"max_accuracy_drop", "max_nll_increase", "max_brier_increase", "max_ece_increase"}
    if set(limits) != required or not all(
        type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in limits.values()
    ):
        raise ValueError("explicit finite nonnegative quality limits are required")
    if set(student) != set(teacher) or not student:
        raise ValueError("quality groups differ")
    failures = []
    for group, reference in teacher.items():
        actual = student[group]
        for metric in ("accuracy", "nll", "brier", "ece"):
            if not math.isfinite(actual[metric]) or not math.isfinite(reference[metric]):
                raise ValueError("non-finite quality metric")
            degradation = (reference[metric] - actual[metric]) if metric == "accuracy" else (
                actual[metric] - reference[metric]
            )
            key = "max_accuracy_drop" if metric == "accuracy" else f"max_{metric}_increase"
            if degradation > limits[key] + 1e-12:
                failures.append({"group": group, "metric": metric, "degradation": degradation,
                                 "limit": limits[key]})
    return {"passed": not failures, "failures": failures}


def _positive_int(config: dict, key: str) -> int:
    value = config[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def make_optimizer(
    parameters: list[nn.Parameter], *, learning_rate: float
) -> torch.optim.AdamW:
    """AdamW without the CUDA foreach tensor-list peak-memory path."""
    if not parameters:
        raise ValueError("no trainable parameters")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=0.0,
        foreach=False,
    )


def run_recovery(
    model: nn.Module, controller: TernaryController, train: Cache, validation: Cache,
    output: Path, config: dict, *, metadata: dict[str, Any],
) -> dict[str, Any]:
    if train.role != "train" or validation.role != "validation":
        raise ValueError("training requires train and validation cache roles")
    assert_disjoint(train, validation)
    if controller.model is not model or controller.loras or any(
        q.frozen for q in controller.targets.values()
    ):
        raise ValueError("recovery must start with a fresh trainable ternary student")
    for key in ("qat_steps", "lora_steps", "evaluation_interval", "lora_rank"):
        _positive_int(config, key)
    for key in ("learning_rate", "lora_learning_rate", "lora_alpha"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"invalid {key}")
    teacher_metrics, _ = evaluate(None, validation.records)
    quality_gate(teacher_metrics, teacher_metrics, config["limits"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    seed = config["seed"]
    if type(seed) is not int:
        raise ValueError("seed must be integer")
    torch.manual_seed(seed)
    randomizer = random.Random(seed)
    with controller.cached_evaluation_codes():
        initial_metrics, initial_logits = evaluate(model, validation.records)
    run_meta = metadata | {"teacher_id": train.teacher_id, "config": config,
                           "train_source_sha256": train.source_sha256,
                           "validation_source_sha256": validation.source_sha256,
                           "split_guard": split_identities(train, validation)}
    write_json(output / "run.json", run_meta | {"inventory": controller.inventory()})
    history = []

    def stage(
        name: str,
        steps: int,
        learning_rate: float,
        *,
        initial_evaluation: tuple[dict, list[list[float]]] | None = None,
    ) -> tuple[Path, dict]:
        # eval mode freezes dropout/BN buffers; gradients remain enabled.
        model.eval()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("no trainable parameters")
        optimizer = make_optimizer(trainable, learning_rate=learning_rate)
        best_path = None
        best_score = None
        best_logits = None
        order = list(range(len(train.records)))
        last_loss = None
        for step in range(steps + 1):
            if step:
                if (step - 1) % len(order) == 0:
                    randomizer.shuffle(order)
                record = train.records[order[(step - 1) % len(order)]]
                optimizer.zero_grad(set_to_none=True)
                logits = model(record)
                loss = distillation_loss(
                    logits, torch.tensor(record.teacher_logits, device=logits.device),
                    record.target,
                    temperature=config.get("temperature", 1.0),
                    ce_weight=config.get("ce_weight", 1.0),
                    brier_weight=config.get("brier_weight", 0.2),
                )
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                loss.backward()
                if not any(p.grad is not None for p in trainable):
                    raise ValueError("no gradients reached the student")
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                last_loss = float(loss.detach())
            if step % config["evaluation_interval"] and step != steps:
                continue
            if step == 0 and initial_evaluation is not None:
                metrics, current_logits = initial_evaluation
                gate = quality_gate(metrics, teacher_metrics, config["limits"])
                score = (not gate["passed"], metrics["all"]["nll"])
                history.append({"stage": name, "step": step, "loss": last_loss,
                                "metrics": metrics, "gate": gate})
                if best_score is None or score < best_score:
                    destination = output / f"{name}-step-{step:06d}"
                    save_bundle(
                        controller, destination, run_meta | {"stage": name, "step": step}
                    )
                    best_path, best_score, best_logits = destination, score, current_logits
                continue
            with controller.cached_evaluation_codes():
                metrics, current_logits = evaluate(model, validation.records)
                gate = quality_gate(metrics, teacher_metrics, config["limits"])
                # Any passing checkpoint outranks every failing one; NLL breaks ties.
                score = (not gate["passed"], metrics["all"]["nll"])
                history.append({"stage": name, "step": step, "loss": last_loss,
                                "metrics": metrics, "gate": gate})
                if best_score is None or score < best_score:
                    destination = output / f"{name}-step-{step:06d}"
                    save_bundle(
                        controller, destination, run_meta | {"stage": name, "step": step}
                    )
                    best_path, best_score, best_logits = destination, score, current_logits
        assert best_path is not None and best_logits is not None
        load_bundle(controller, best_path)
        selected_metrics, reloaded_logits = evaluate(model, validation.records)
        error = max(abs(a - b) for xs, ys in zip(best_logits, reloaded_logits, strict=True)
                    for a, b in zip(xs, ys, strict=True))
        if error > config.get("reload_atol", 1e-6):
            raise ValueError(f"export/reload prediction mismatch: {error}")
        return best_path, {"metrics": selected_metrics, "reload_max_abs_error": error}

    selected, result = stage(
        "qat",
        config["qat_steps"],
        config["learning_rate"],
        initial_evaluation=(initial_metrics, initial_logits),
    )
    gate = quality_gate(result["metrics"], teacher_metrics, config["limits"])
    lora_used = not gate["passed"]
    if lora_used:
        # load_bundle already froze the selected codes, scales, and other base tensors.
        controller.add_lora(config["lora_rank"], config["lora_alpha"])
        selected, result = stage("lora", config["lora_steps"], config["lora_learning_rate"])
        gate = quality_gate(result["metrics"], teacher_metrics, config["limits"])
    total_bytes = sum(p.stat().st_size for p in selected.iterdir())
    size_passed = total_bytes <= config.get("max_artifact_bytes", math.inf)
    report = result | {
        "status": "PASS" if gate["passed"] and size_passed else "FAIL", "gate": gate,
        "size_gate_passed": size_passed,
        "lora_used": lora_used, "selected": selected.name,
        "teacher_metrics": teacher_metrics, "initial_ternary_metrics": initial_metrics,
        "history": history, "final_evaluation_used": False,
        "total_bytes": total_bytes,
    }
    write_json(output / "result.json", report)
    return report
