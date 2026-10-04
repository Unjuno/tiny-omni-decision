from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .corpus import (
    ValidationCheckpointSelector,
    comparison_deltas,
    file_sha256,
    macro_metrics,
    validate_training_inputs,
    validation_selection_score,
)
from .decision import decision_loss, normalize_probabilities, option_logits_from_vocab
from .decision_math import brier_score, expected_calibration_error, negative_log_likelihood
from .experiment import ExperimentManifest, append_experiment_event
from .io import load_structured_file
from .schema import BaseModelManifest, DecisionExample
from .training import (
    DecisionTrainingConfig,
    aggregate_training_window,
    decision_training_config,
    deterministic_sample_order,
    deterministic_validation_subset,
    learning_rate_multiplier,
    output_record,
    processor_inputs_for_example,
    project_runtime_seconds,
    resolve_decoder_lora_targets,
    sampling_accounting,
)


def _read_examples(path: Path) -> list[DecisionExample]:
    with path.open(encoding="utf-8") as handle:
        return [DecisionExample.model_validate_json(line) for line in handle if line.strip()]


def _eligible(
    examples: list[DecisionExample], modalities: set[str]
) -> tuple[list[DecisionExample], dict[str, int]]:
    eligible: list[DecisionExample] = []
    skipped: Counter[str] = Counter()
    for example in examples:
        if example.modality not in modalities:
            skipped[f"modality_not_selected:{example.modality}"] += 1
            continue
        if example.modality != "text" and any(not item.path for item in example.media):
            skipped[f"media_unresolved:{example.modality}"] += 1
            continue
        eligible.append(example)
    return eligible, dict(sorted(skipped.items()))


def _move_inputs(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }


def _load_pretrained_base(model_loader: Any, repo_id: str, *, revision: str, **kwargs: Any) -> Any:
    """Use bounded-memory safetensors reads on Windows, restoring the loader after use."""
    if sys.platform != "win32":
        return model_loader(repo_id, revision=revision, **kwargs)

    modeling_utils = importlib.import_module("transformers.modeling_utils")
    original_safe_open = modeling_utils.safe_open

    def windows_pread_safe_open(*args: Any, **open_kwargs: Any) -> Any:
        open_kwargs["backend"] = "pread"
        return original_safe_open(*args, **open_kwargs)

    modeling_utils.safe_open = windows_pread_safe_open
    try:
        return model_loader(repo_id, revision=revision, **kwargs)
    finally:
        modeling_utils.safe_open = original_safe_open


def _forward_decision(
    model: Any,
    processor: Any,
    example: DecisionExample,
    *,
    data_root: Path,
    max_sequence_length: int,
    video_num_frames: int,
) -> tuple[Any, int]:
    import torch

    inputs, option_ids, position, target = processor_inputs_for_example(
        processor, example, data_root=data_root, video_num_frames=video_num_frames
    )
    if inputs["input_ids"].shape[-1] > max_sequence_length:
        raise ValueError(
            f"{example.id}: processor sequence length {inputs['input_ids'].shape[-1]} "
            f"exceeds configured max {max_sequence_length}"
        )
    device = model.get_input_embeddings().weight.device
    output = model(**_move_inputs(inputs, device), use_cache=False)
    logits = option_logits_from_vocab(output.logits[0, position], option_ids)
    if not torch.isfinite(logits).all():
        raise ValueError(f"{example.id}: option logits are non-finite")
    return logits, target


def _evaluate(
    model: Any,
    processor: Any,
    examples: list[DecisionExample],
    *,
    data_root: Path,
    config: DecisionTrainingConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    predictions = []
    groups: dict[str, list[tuple[list[float], int]]] = defaultdict(list)
    with torch.no_grad():
        for example in examples:
            logits, target = _forward_decision(
                model,
                processor,
                example,
                data_root=data_root,
                max_sequence_length=config.max_sequence_length,
                video_num_frames=config.video_num_frames,
            )
            probabilities = normalize_probabilities(logits).cpu().tolist()
            values = logits.float().cpu().tolist()
            predictions.append(output_record(example, values, probabilities))
            groups["all"].append((probabilities, target))
            groups[f"modality:{example.modality}"].append((probabilities, target))
            groups[f"source:{example.source}"].append((probabilities, target))

    def measure(items: list[tuple[list[float], int]]) -> dict[str, float | int]:
        probs = [item[0] for item in items]
        targets = [item[1] for item in items]
        return {
            "count": len(items),
            "accuracy": sum(
                max(range(len(row)), key=row.__getitem__) == target for row, target in items
            )
            / len(items),
            "nll": negative_log_likelihood(probs, targets),
            "brier": brier_score(probs, targets),
            "ece": expected_calibration_error(probs, targets, config.ece_bins),
            "mean_confidence": sum(max(row) for row in probs) / len(probs),
        }

    return {key: measure(values) for key, values in sorted(groups.items())}, predictions


def publish_best_adapter(temporary_best: Path, best_path: Path) -> None:
    """Publish the selected adapter while tolerating transient Windows file locks."""
    backup_path = best_path.with_name(f".best-backup-{uuid.uuid4().hex}")

    def replace_with_retry(source: Path, destination: Path) -> None:
        for attempt in range(6):
            try:
                source.replace(destination)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(min(0.05 * (2**attempt), 1.0))

    if best_path.exists():
        replace_with_retry(best_path, backup_path)
    try:
        replace_with_retry(temporary_best, best_path)
    except Exception:
        if backup_path.exists() and not best_path.exists():
            replace_with_retry(backup_path, best_path)
        raise
    if backup_path.exists():
        shutil.rmtree(backup_path, ignore_errors=True)


def _run_training_impl(
    *,
    train_path: Path,
    validation_path: Path,
    config_path: Path,
    output_dir: Path,
    experiment_id: str,
    reference_adapter_path: Path | None = Path("artifacts/tiny-omni-decision-teacher-v0/best"),
    seed_override: int | None = None,
    max_train_examples: int | None = None,
    max_steps: int | None = None,
    gradient_accumulation_steps: int | None = None,
    checkpoint_interval: int | None = None,
    evaluation_interval: int | None = None,
    resume_from: Path | None = None,
    modalities: set[str] | None = None,
    tiny_overfit: bool = False,
    model_manifest_path: Path = Path("manifests/base-model.example.yaml"),
) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    raw = load_structured_file(config_path)
    config = decision_training_config(raw)
    reference_teacher_id = str(raw.get("reference_teacher_id", "tiny-omni-decision-teacher-v0"))
    if raw.get("reference_teacher_id") and (
        reference_adapter_path is None or not reference_adapter_path.is_dir()
    ):
        raise ValueError(
            f"configured reference adapter is missing for {reference_teacher_id}: "
            f"{reference_adapter_path}"
        )
    overrides = {
        "seed": seed_override,
        "max_train_examples": max_train_examples,
        "max_steps": max_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "checkpoint_interval": checkpoint_interval,
        "evaluation_interval": evaluation_interval,
    }
    config = config.model_copy(
        update={key: value for key, value in overrides.items() if value is not None}
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Decision LoRA training requires the local CUDA device")
    if tiny_overfit:
        config = config.model_copy(
            update={"max_train_examples": min(config.max_train_examples, 16)}
        )
        if config.max_steps < 8:
            config = config.model_copy(update={"max_steps": 8})
    selected_modalities = modalities or {"text", "image", "audio", "video"}
    if not selected_modalities <= {"text", "image", "audio", "video"}:
        raise ValueError("modalities must be selected from text,image,audio,video")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    train_raw = _read_examples(train_path)
    validation_raw = _read_examples(validation_path)
    from .dataset import audit_license, check_train_eval_splits

    for example in train_raw:
        policy, unresolved = audit_license(
            example.provenance.license,
            commercial_use=example.provenance.commercial_use,
            derivative_model_training_allowed=example.provenance.derivative_model_training_allowed,
            redistribution_allowed=example.provenance.redistribution_allowed,
            media_redistribution_allowed=example.provenance.media_redistribution_allowed,
            has_media=bool(example.media),
            trust_status=example.provenance.trust_status,
        )
        if policy != "ALLOW":
            raise ValueError(
                f"training example is not ALLOW ({policy}): {example.id}: {unresolved}"
            )
    result = check_train_eval_splits(train_raw, validation_raw)
    if result["status"] != "disjoint":
        raise ValueError(f"train/validation contamination detected: {result}")

    train_ready, train_skipped = _eligible(train_raw, selected_modalities)
    validation_ready, validation_skipped = _eligible(validation_raw, selected_modalities)
    if not train_ready or not validation_ready:
        raise ValueError(
            "no locally materialized training or independent validation samples remain"
        )
    train_order, _ = deterministic_sample_order(
        train_ready,
        seed=config.seed,
        limit=config.max_train_examples,
        modality_weights=config.modality_weights,
        source_weights=config.source_weights,
        max_sample_repeats=config.max_sample_repeats,
    )
    validation_subset = deterministic_validation_subset(
        validation_ready,
        seed=17,
        limit=min(config.selection_eval_examples, len(validation_ready)),
    )
    if tiny_overfit:
        train_order = train_order[: min(len(train_order), 8)]
    elif len(train_order) < config.max_steps * config.gradient_accumulation_steps:
        raise ValueError(
            "training schedule is shorter than the configured optimization run; "
            "increase the corpus/reuse cap or reduce max_steps"
        )

    manifest = BaseModelManifest.model_validate(load_structured_file(model_manifest_path))
    processor = AutoProcessor.from_pretrained(
        manifest.processor_repo_id or manifest.repo_id,
        revision=manifest.processor_revision,
    )
    base = _load_pretrained_base(
        AutoModelForMultimodalLM.from_pretrained,
        manifest.repo_id,
        revision=manifest.revision,
        dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    data_root = train_path.parent.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_started = time.monotonic()
    validation_baseline_metrics, validation_baseline_predictions = _evaluate(
        base, processor, validation_subset, data_root=data_root, config=config
    )
    validation_baseline_eval_seconds = time.monotonic() - validation_started
    validation_baseline_metrics["macro"] = macro_metrics(validation_baseline_metrics)
    (output_dir / "validation-baseline-evaluation.json").write_text(
        json.dumps(validation_baseline_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "validation-baseline-predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in validation_baseline_predictions),
        encoding="utf-8",
    )
    reference_validation_metrics = None
    reference_validation_eval_seconds = None
    if reference_adapter_path is not None and reference_adapter_path.is_dir():
        from peft import PeftModel

        reference = PeftModel.from_pretrained(base, reference_adapter_path, is_trainable=False)
        reference.eval()
        reference_started = time.monotonic()
        reference_validation_metrics, _ = _evaluate(
            reference, processor, validation_subset, data_root=data_root, config=config
        )
        reference_validation_eval_seconds = time.monotonic() - reference_started
        reference_validation_metrics["macro"] = macro_metrics(reference_validation_metrics)
        base = reference.unload()
        del reference
        gc.collect()
        base.eval()
    tiny_before = None
    if tiny_overfit:
        tiny_train_metrics, _ = _evaluate(
            base, processor, train_order, data_root=data_root, config=config
        )
        tiny_before = tiny_train_metrics["all"]["nll"] + config.brier_weight * tiny_train_metrics[
            "all"
        ]["brier"]
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    targets = resolve_decoder_lora_targets(base, policy=config.lora_target_policy)
    if resume_from:
        model = PeftModel.from_pretrained(base, resume_from, is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                use_rslora=config.use_rslora,
                target_modules=targets,
                task_type="CAUSAL_LM",
            ),
        )
    actual_targets = sorted(
        name.removeprefix("base_model.model.")
        for name, module in model.named_modules()
        if hasattr(module, "lora_A")
    )
    if actual_targets != targets:
        raise RuntimeError("PEFT target paths did not match the loaded pinned architecture")
    adapter_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen_parameters = [
        parameter for name, parameter in model.named_parameters() if ".lora_" not in name
    ]
    if not adapter_parameters or any(parameter.requires_grad for parameter in frozen_parameters):
        raise RuntimeError("base-freeze / LoRA-trainable invariant failed")
    base_versions = [parameter._version for parameter in frozen_parameters]
    optimizer = torch.optim.AdamW(adapter_parameters, lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate_multiplier(
            step,
            total_steps=config.max_steps,
            scheduler=config.lr_scheduler,
            warmup_ratio=config.warmup_ratio,
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    train_hash = file_sha256(str(train_path))
    validation_hash = file_sha256(str(validation_path))
    config_hash = file_sha256(str(config_path))
    if resume_from and (resume_from / "optimizer.pt").is_file():
        optimizer.load_state_dict(
            torch.load(resume_from / "optimizer.pt", map_location="cpu", weights_only=True)
        )
    if resume_from and (resume_from / "scheduler.pt").is_file():
        lr_scheduler.load_state_dict(
            torch.load(resume_from / "scheduler.pt", map_location="cpu", weights_only=True)
        )
    model.train()
    consumed: Counter[str] = Counter()
    consumed_examples: list[DecisionExample] = []
    losses: list[float] = []
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    sample_index = 0
    global_step = 0
    selector = ValidationCheckpointSelector(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )
    best_selection_loss = float("inf")
    best_step = 0
    best_validation_predictions: list[dict[str, Any]] | None = None
    if resume_from and (resume_from / "trainer-state.json").is_file():
        saved_state = json.loads((resume_from / "trainer-state.json").read_text(encoding="utf-8"))
        expected_hashes = {
            "train_rows_sha256": train_hash,
            "validation_rows_sha256": validation_hash,
            "config_sha256": config_hash,
        }
        for key, expected in expected_hashes.items():
            if saved_state.get(key) != expected:
                raise ValueError(f"resume checkpoint {key} does not match current run input")
        global_step = int(saved_state["global_step"])
        sample_index = int(saved_state.get("sample_index", 0))
        consumed.update(saved_state.get("consumed", {}))
        losses.extend(float(value) for value in saved_state.get("losses", []))
        history.extend(saved_state.get("history", []))
        best_selection_loss = float(saved_state.get("best_selection_loss", float("inf")))
        best_step = int(saved_state.get("best_step", 0))
        selector.best_score = best_selection_loss
        selector.best_step = best_step
        selector.evaluations_without_improvement = int(
            saved_state.get("evaluations_without_improvement", 0)
        )
        consumed_examples.extend(
            train_order[index % len(train_order)]
            for index in range(sample_index)
        )
        best_path = output_dir / "best"
        if best_step and not best_path.is_dir():
            raise ValueError("resume checkpoint refers to a missing best adapter directory")
        best_predictions_path = output_dir / "best-validation-predictions.jsonl"
        if best_step and best_predictions_path.is_file():
            best_validation_predictions = [
                json.loads(line)
                for line in best_predictions_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
    else:
        best_path = output_dir / "best"

    history_path = output_dir / "training-history.jsonl"

    def state_record(step: int) -> dict[str, Any]:
        return {
            "global_step": step,
            "sample_index": sample_index,
            "best_selection_loss": best_selection_loss,
            "best_step": best_step,
            "evaluations_without_improvement": selector.evaluations_without_improvement,
            "consumed": dict(consumed),
            "losses": losses,
            "history": history,
            "train_rows_sha256": train_hash,
            "validation_rows_sha256": validation_hash,
            "config_sha256": config_hash,
            "base_revision": manifest.revision,
            "experiment_id": experiment_id,
        }

    while global_step < config.max_steps:
        step_started = time.monotonic()
        step_total_losses: list[float] = []
        step_cross_entropies: list[float] = []
        step_briers: list[float] = []
        step_composition: Counter[str] = Counter()
        step_correct_by_modality: Counter[str] = Counter()
        step_count_by_modality: Counter[str] = Counter()
        step_ce_by_modality: dict[str, list[float]] = defaultdict(list)
        for _ in range(config.gradient_accumulation_steps):
            example = train_order[sample_index % len(train_order)]
            sample_index += 1
            consumed_examples.append(example)
            batch_key = f"{example.modality}:{example.source}"
            consumed[batch_key] += 1
            step_composition[batch_key] += 1
            logits, target = _forward_decision(
                model,
                processor,
                example,
                data_root=data_root,
                max_sequence_length=config.max_sequence_length,
                video_num_frames=config.video_num_frames,
            )
            loss, cross_entropy, brier = decision_loss(
                logits.unsqueeze(0),
                torch.tensor([target], device=logits.device),
                brier_weight=config.brier_weight,
                cross_entropy_weight=config.cross_entropy_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at sample {example.id}")
            (loss / config.gradient_accumulation_steps).backward()
            loss_value = float(loss.detach())
            losses.append(loss_value)
            step_total_losses.append(loss_value)
            step_cross_entropies.append(float(cross_entropy.detach()))
            step_briers.append(float(brier.detach()))
            step_correct_by_modality[example.modality] += int(
                int(torch.argmax(logits).item()) == target
            )
            step_count_by_modality[example.modality] += 1
            step_ce_by_modality[example.modality].append(float(cross_entropy.detach()))
        for parameter in adapter_parameters:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Decision LoRA has a missing or non-finite gradient")
        if any(parameter.grad is not None for parameter in frozen_parameters):
            raise RuntimeError("frozen base received gradients")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(adapter_parameters, config.max_gradient_norm)
        )
        if not torch.isfinite(torch.tensor(gradient_norm)):
            raise RuntimeError("Decision LoRA gradient norm is non-finite")
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if [parameter._version for parameter in frozen_parameters] != base_versions:
            raise RuntimeError("a frozen base parameter changed during training")
        step_record = {
            "step": global_step,
            "cross_entropy": sum(step_cross_entropies) / len(step_cross_entropies),
            "train_accuracy": sum(step_correct_by_modality.values())
            / sum(step_count_by_modality.values()),
            "train_accuracy_by_modality": {
                modality: step_correct_by_modality[modality] / count
                for modality, count in sorted(step_count_by_modality.items())
            },
            "train_ce_by_modality": {
                modality: sum(step_ce_by_modality[modality]) / len(step_ce_by_modality[modality])
                for modality in sorted(step_ce_by_modality)
            },
            "train_correct_by_modality": dict(sorted(step_correct_by_modality.items())),
            "train_examples_by_modality": dict(sorted(step_count_by_modality.items())),
            "train_examples": sum(step_count_by_modality.values()),
            "microbatches": config.gradient_accumulation_steps,
            "brier": sum(step_briers) / len(step_briers),
            "total_loss": sum(step_total_losses) / len(step_total_losses),
            "step_seconds": time.monotonic() - step_started,
            "learning_rate": learning_rate_used,
            "gradient_norm": gradient_norm,
            "sample_composition": dict(sorted(step_composition.items())),
        }
        history.append(step_record)
        with history_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(step_record) + "\n")

        selection_metrics = None
        if global_step % config.evaluation_interval == 0 or global_step == config.max_steps:
            validation_started = time.monotonic()
            selection_metrics, selection_predictions = _evaluate(
                model, processor, validation_subset, data_root=data_root, config=config
            )
            validation_eval_seconds = time.monotonic() - validation_started
            selection_metrics["macro"] = macro_metrics(selection_metrics)
            score = validation_selection_score(selection_metrics)
            selection_metrics["selection_score"] = score
            previous_validation = next(
                (item for item in reversed(history) if "validation" in item), None
            )
            prior_validation_step = (
                int(previous_validation["step"]) if previous_validation is not None else 0
            )
            training_window = [
                item
                for item in history
                if "step_seconds" in item and int(item["step"]) > prior_validation_step
            ]
            training_metrics = aggregate_training_window(training_window)
            overfit_signals = {
                "train_loss_falling": bool(
                    previous_validation
                    and training_metrics["train_ce"] < previous_validation["train"]["train_ce"]
                ),
                "validation_nll_rising": bool(
                    previous_validation
                    and selection_metrics["macro"]["macro_nll"]
                    > previous_validation["validation"]["macro"]["macro_nll"]
                ),
                "train_accuracy_rising_validation_stalled": bool(
                    previous_validation
                    and training_metrics["train_accuracy"]
                    > previous_validation["train"]["train_accuracy"]
                    and selection_metrics["macro"]["macro_accuracy"]
                    <= previous_validation["validation"]["macro"]["macro_accuracy"]
                ),
                "weak_modality_degraded": bool(
                    previous_validation
                    and selection_metrics["macro"]["minimum_modality_accuracy"]
                    < previous_validation["validation"]["macro"]["minimum_modality_accuracy"]
                ),
            }
            validation_record = {
                "step": global_step,
                "train": training_metrics,
                "validation": selection_metrics,
                "validation_eval_seconds": validation_eval_seconds,
                "overfit_signals": overfit_signals,
            }
            history.append(validation_record)
            improved = selector.observe(global_step, selection_metrics)
            best_selection_loss = selector.best_score
            best_step = selector.best_step
            (output_dir / f"validation-step-{global_step}.json").write_text(
                json.dumps(validation_record, indent=2), encoding="utf-8"
            )
            if improved:
                best_validation_predictions = selection_predictions
                temporary_best = output_dir / "best.tmp"
                if temporary_best.exists():
                    shutil.rmtree(temporary_best)
                model.save_pretrained(temporary_best)
                publish_best_adapter(temporary_best, best_path)
                (output_dir / "best-checkpoint.json").write_text(
                    json.dumps(
                        {
                            "step": best_step,
                            "selection_score": best_selection_loss,
                            "selection_rule": "macro NLL + 0.2 macro Brier + 0.1 macro ECE "
                            "- 0.25 macro accuracy - 0.25 minimum modality accuracy",
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (output_dir / "best-validation-predictions.jsonl").write_text(
                    "".join(json.dumps(item) + "\n" for item in selection_predictions),
                    encoding="utf-8",
                )

        if global_step % config.checkpoint_interval == 0 or global_step == config.max_steps:
            checkpoint_path = checkpoints_dir / f"step-{global_step:06d}"
            checkpoint_path.mkdir(exist_ok=True)
            model.save_pretrained(checkpoint_path)
            torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.pt")
            torch.save(lr_scheduler.state_dict(), checkpoint_path / "scheduler.pt")
            (checkpoint_path / "trainer-state.json").write_text(
                json.dumps(state_record(global_step)), encoding="utf-8"
            )
        if global_step % config.evaluation_interval == 0 or global_step == config.max_steps:
            (output_dir / "trainer-state.json").write_text(
                json.dumps(state_record(global_step)), encoding="utf-8"
            )
        model.train()
        if selector.should_stop:
            break
    if best_step == 0:
        raise RuntimeError("training ended without selecting a best validation checkpoint")

    # Reload only the validation-selected adapter; the sealed audit is not read here.
    model.eval()
    unloaded_base = model.unload()
    del model
    gc.collect()
    reloaded = PeftModel.from_pretrained(unloaded_base, best_path, is_trainable=False).eval()
    final_validation_started = time.monotonic()
    reloaded_validation_metrics, reloaded_validation_predictions = _evaluate(
        reloaded, processor, validation_subset, data_root=data_root, config=config
    )
    final_validation_eval_seconds = time.monotonic() - final_validation_started
    reloaded_validation_metrics["macro"] = macro_metrics(reloaded_validation_metrics)
    if best_validation_predictions is None or [
        record["option_probabilities"] for record in best_validation_predictions
    ] != [record["option_probabilities"] for record in reloaded_validation_predictions]:
        raise RuntimeError("best checkpoint save/reload validation predictions changed")
    tiny_after = None
    if tiny_overfit:
        tiny_train_metrics, _ = _evaluate(
            reloaded, processor, train_order, data_root=data_root, config=config
        )
        tiny_after = tiny_train_metrics["all"]["nll"] + config.brier_weight * tiny_train_metrics[
            "all"
        ]["brier"]
        if tiny_after >= tiny_before:
            raise RuntimeError(f"tiny-overfit loss did not decrease: {tiny_before} -> {tiny_after}")

    torch.cuda.synchronize()
    max_vram = torch.cuda.max_memory_allocated()
    end_at = datetime.now(UTC)
    wall_seconds = time.monotonic() - started_clock
    usage = sampling_accounting(consumed_examples)
    modality_consumption: Counter[str] = Counter()
    for key, count in consumed.items():
        modality_consumption[key.split(":", 1)[0]] += count
    best_weights_path = best_path / "adapter_model.safetensors"
    if not best_weights_path.is_file():
        best_weights_path = best_path / "adapter_model.bin"
    checkpoint_hashes = {
        path.parent.name: file_sha256(str(path))
        for path in sorted(checkpoints_dir.glob("step-*/adapter_model.safetensors"))
    }
    corpus_manifest_path = train_path.parent / "corpus-manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    training_seconds = sum(
        float(item["step_seconds"]) for item in history if "step_seconds" in item
    )
    validation_points = [item for item in history if "validation" in item]
    validation_eval_seconds = validation_baseline_eval_seconds + float(
        reference_validation_eval_seconds or 0.0
    ) + sum(float(item["validation_eval_seconds"]) for item in validation_points)
    validation_eval_seconds += final_validation_eval_seconds
    selected_seconds_per_step = training_seconds / max(global_step, 1)
    scheduled_eval_times = [
        float(item["validation_eval_seconds"]) for item in validation_points
    ]
    mean_scheduled_eval_seconds = (
        sum(scheduled_eval_times) / len(scheduled_eval_times) if scheduled_eval_times else 0.0
    )
    setup_validation_seconds = validation_baseline_eval_seconds + float(
        reference_validation_eval_seconds or 0.0
    )
    fixed_runtime_overhead_seconds = max(
        0.0, wall_seconds - training_seconds - validation_eval_seconds
    )
    adapter_size_bytes = best_weights_path.stat().st_size
    trainable_parameter_count = sum(parameter.numel() for parameter in adapter_parameters)
    total_parameter_count = sum(parameter.numel() for parameter in reloaded.parameters())
    overfit_signal_points = [
        item["overfit_signals"] for item in validation_points if "overfit_signals" in item
    ]
    sealed_audit_sha256 = corpus_manifest.get("sealed_audit_sha256")
    validation_reference_deltas = (
        comparison_deltas(reference_validation_metrics, reloaded_validation_metrics)
        if reference_validation_metrics is not None
        else None
    )
    metadata = {
        "teacher_id": str(raw.get("teacher_id", "tiny-omni-decision-teacher-v1-candidate")),
        "experiment_id": experiment_id,
        "artifact_role": "validation_selected_experiment_candidate",
        "base_repo_id": manifest.repo_id,
        "base_revision": manifest.revision,
        "seed": config.seed,
        "config": config.model_dump(mode="json"),
        "lora_target_policy": config.lora_target_policy,
        "lora_rank": config.lora_rank,
        "use_rslora": config.use_rslora,
        "target_modules": targets,
        "base_weights_sha256": next(
            (item["sha256"] for item in manifest.files if item.get("path") == "model.safetensors"),
            None,
        ),
        "train_rows_sha256": train_hash,
        "validation_rows_sha256": validation_hash,
        "sealed_audit_sha256": sealed_audit_sha256,
        "corpus_manifest_sha256": file_sha256(str(corpus_manifest_path)),
        "training_config_sha256": config_hash,
        "train_examples_available": len(train_ready),
        "train_examples_selected": len(train_order),
        "validation_examples_available": len(validation_ready),
        "validation_examples_measured": len(validation_subset),
        "skipped_train_examples": train_skipped,
        "skipped_validation_examples": validation_skipped,
        "actual_train_consumption_by_modality_source": dict(sorted(consumed.items())),
        "actual_train_consumption_by_modality": dict(sorted(modality_consumption.items())),
        "sample_accounting": usage,
        "train_loss_mean": sum(losses) / len(losses),
        "tiny_overfit_nll_before": tiny_before,
        "tiny_overfit_nll_after": tiny_after,
        "validation_base_metrics": validation_baseline_metrics,
        "validation_reference_teacher_id": reference_teacher_id,
        "validation_reference_metrics": reference_validation_metrics,
        "validation_candidate_metrics": reloaded_validation_metrics,
        "validation_deltas_candidate_minus_base": comparison_deltas(
            validation_baseline_metrics, reloaded_validation_metrics
        ),
        "validation_teacher_v0_metrics": (
            reference_validation_metrics
            if reference_teacher_id == "tiny-omni-decision-teacher-v0"
            else None
        ),
        "validation_teacher_v1_candidate_metrics": reloaded_validation_metrics,
        "validation_deltas_v1_candidate_minus_base": comparison_deltas(
            validation_baseline_metrics, reloaded_validation_metrics
        ),
        "validation_deltas_candidate_minus_reference": validation_reference_deltas,
        "validation_deltas_v1_candidate_minus_teacher_v0": (
            validation_reference_deltas
            if reference_teacher_id == "tiny-omni-decision-teacher-v0"
            else None
        ),
        "validation_baseline_metrics": validation_baseline_metrics,
        "validation_best_metrics": reloaded_validation_metrics,
        "validation_best_selection_score": best_selection_loss,
        "validation_macro_metrics": reloaded_validation_metrics["macro"],
        "validation_learning_curve": validation_points,
        "overfit_signal_points": overfit_signal_points,
        "best_checkpoint_step": best_step,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "stopped_early": global_step < config.max_steps,
        "best_adapter_path": "best",
        "step_checkpoint_sha256": checkpoint_hashes,
        "best_adapter_sha256": file_sha256(str(best_weights_path)),
        "adapter_config_sha256": file_sha256(str(best_path / "adapter_config.json")),
        "adapter_size_bytes": adapter_size_bytes,
        "trainable_parameter_count": trainable_parameter_count,
        "projector_trainable_parameter_count": 0,
        "total_parameter_count": total_parameter_count,
        "expected_merge_impact": {
            "adapter_trainable_parameters_absorbed_into_base": trainable_parameter_count,
            "additional_projector_parameters": 0,
            "separate_runtime_adapter_after_merge": False,
            "ternary_conversion_performed": False,
        },
        "global_steps": global_step,
        "optimizer_microbatches": sample_index,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler": config.lr_scheduler,
        "warmup_ratio": config.warmup_ratio,
        "max_gradient_norm": config.max_gradient_norm,
        "mean_seconds_per_optimizer_step": selected_seconds_per_step,
        "training_seconds": training_seconds,
        "validation_evaluation_seconds": validation_eval_seconds,
        "projected_seconds_by_budget": {
            str(steps): selected_seconds_per_step * steps for steps in (512, 1024, 2048, 4096)
        },
        "projected_total_runtime_seconds_by_budget": {
            str(steps): project_runtime_seconds(
                seconds_per_optimizer_step=selected_seconds_per_step,
                optimizer_steps=steps,
                evaluation_interval=config.evaluation_interval,
                mean_scheduled_evaluation_seconds=mean_scheduled_eval_seconds,
                setup_evaluation_seconds=setup_validation_seconds,
                final_evaluation_seconds=final_validation_eval_seconds,
                other_overhead_seconds=fixed_runtime_overhead_seconds,
            )
            for steps in (512, 1024, 2048, 4096)
        },
        "projected_fixed_runtime_overhead_seconds": fixed_runtime_overhead_seconds,
        "sampling_mixture": {
            "modality_weights": config.modality_weights,
            "source_weights": config.source_weights,
            "max_sample_repeats": config.max_sample_repeats,
            "scheduler": (
                "weighted modality round robin; source weights apply within modality; "
                "sample reuse is bounded by max_sample_repeats"
            ),
            "consumed_by_modality_source": dict(sorted(consumed.items())),
            **usage,
        },
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "max_allocated_vram_bytes": max_vram,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
        "safetensors_backend": "pread" if sys.platform == "win32" else "transformers-default",
        "peft_version": __import__("peft").__version__,
        "accelerate_version": importlib.metadata.version("accelerate"),
        "python_version": __import__("platform").python_version(),
        "checkpoint_reload_verified": True,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": end_at.isoformat(),
        "wall_seconds": wall_seconds,
        "rental_provider": "local workstation",
        "rental_gpu_hourly_usd": 0.0,
        "rental_total_usd": 0.0,
        "merge_export_performed": False,
    }
    (output_dir / "run-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "validation-predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in reloaded_validation_predictions),
        encoding="utf-8",
    )
    (output_dir / "validation-teacher-evaluation.json").write_text(
        json.dumps(reloaded_validation_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "learning-curves.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in validation_points), encoding="utf-8"
    )
    (output_dir / "pip-freeze.txt").write_text(
        "\n".join(
            sorted(
                f"{distribution.metadata['Name']}=={distribution.version}"
                for distribution in importlib.metadata.distributions()
                if distribution.metadata.get("Name")
            )
        )
        + "\n",
        encoding="utf-8",
    )
    metadata["environment"] = {
        key: metadata[key]
        for key in (
            "python_version",
            "torch_version",
            "cuda_runtime_version",
            "transformers_version",
            "safetensors_backend",
            "peft_version",
            "accelerate_version",
            "gpu",
            "gpu_total_memory_bytes",
        )
    }
    (output_dir / "run-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_training(**kwargs: Any) -> dict[str, Any]:
    """Log every attempt, require independent validation, and keep audit out of iteration."""
    output_dir = Path(kwargs["output_dir"])
    if output_dir.name == "tiny-omni-decision-teacher-v0":
        raise ValueError("Teacher v0 is immutable; write experiments to a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    ledger_path = output_dir / "experiments.jsonl"
    started_at = datetime.now(UTC).isoformat()
    config_path = Path(kwargs["config_path"])
    model_manifest_path = Path(
        kwargs.get("model_manifest_path", Path("manifests/base-model.example.yaml"))
    )
    start_details: dict[str, Any] = {}
    if config_path.is_file():
        start_details["config_sha256"] = file_sha256(str(config_path))
        try:
            config_raw = load_structured_file(config_path)
            start_details["seed"] = kwargs.get("seed_override")
            if start_details["seed"] is None:
                start_details["seed"] = int(config_raw.get("training", {}).get("seed", 17))
            start_details["sampling_policy"] = config_raw.get("sampling", {})
        except Exception:
            pass
    if model_manifest_path.is_file():
        try:
            base_manifest = BaseModelManifest.model_validate(
                load_structured_file(model_manifest_path)
            )
            start_details["base_repo_id"] = base_manifest.repo_id
            start_details["base_revision"] = base_manifest.revision
        except Exception:
            pass
    for name, key in (
        ("train_path", "train_corpus_sha256"),
        ("validation_path", "validation_corpus_sha256"),
    ):
        value = kwargs.get(name)
        path = Path(value) if value is not None else None
        if path is not None and path.is_file():
            start_details[key] = file_sha256(str(path))
    append_experiment_event(
        ledger_path,
        {
            "experiment_id": experiment_id,
            "status": "started",
            "started_at_utc": started_at,
            "train_path": str(kwargs.get("train_path")),
            "validation_path": str(kwargs.get("validation_path")),
            "config_path": str(kwargs.get("config_path")),
            **start_details,
        },
    )
    try:
        validate_training_inputs(
            kwargs["train_path"],
            kwargs.get("validation_path"),
            evaluation_path=kwargs.get("eval_path"),
        )
        if kwargs.get("eval_path") is not None:
            raise ValueError("evaluation is isolated from normal experiment iteration")
        kwargs.pop("eval_path", None)
        kwargs["experiment_id"] = experiment_id
        kwargs.setdefault(
            "reference_adapter_path", Path("artifacts/tiny-omni-decision-teacher-v0/best")
        )
        result = _run_training_impl(**kwargs)
        manifest = ExperimentManifest(
            experiment_id=experiment_id,
            status="completed",
            seed=int(result["seed"]),
            base_model_repo_id=str(result["base_repo_id"]),
            base_model_revision=str(result["base_revision"]),
            base_model_weights_sha256=result["base_weights_sha256"],
            train_corpus_sha256=str(result["train_rows_sha256"]),
            validation_corpus_sha256=str(result["validation_rows_sha256"]),
            sealed_audit_corpus_sha256=result["sealed_audit_sha256"],
            config_sha256=str(result["training_config_sha256"]),
            sampling_policy=result["sampling_mixture"],
            optimizer_schedule={
                "optimizer_steps": result["global_steps"],
                "gradient_accumulation_steps": result["gradient_accumulation_steps"],
                "learning_rate": result["learning_rate"],
                "early_stopping_patience": result["early_stopping_patience"],
            },
            learning_curve=result["validation_learning_curve"],
            metrics={
                "validation": result["validation_best_metrics"],
                "reference_teacher_id": result["validation_reference_teacher_id"],
                "reference_validation": result["validation_reference_metrics"],
                "teacher_v0_validation": result["validation_teacher_v0_metrics"],
            },
            artifact={
                "best_step": result["best_checkpoint_step"],
                "adapter_size_bytes": result["adapter_size_bytes"],
                "trainable_parameter_count": result["trainable_parameter_count"],
                "projector_trainable_parameter_count": result[
                    "projector_trainable_parameter_count"
                ],
                "best_adapter_sha256": result["best_adapter_sha256"],
                "expected_merge_impact": result["expected_merge_impact"],
            },
            environment={
                "gpu": result["gpu"],
                "gpu_total_memory_bytes": result["gpu_total_memory_bytes"],
                "max_allocated_vram_bytes": result["max_allocated_vram_bytes"],
                "torch_version": result["torch_version"],
                "cuda_runtime_version": result["cuda_runtime_version"],
                "transformers_version": result["transformers_version"],
                "safetensors_backend": result["safetensors_backend"],
                "peft_version": result["peft_version"],
                "accelerate_version": result["accelerate_version"],
            },
        )
        (output_dir / "experiment-manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        result["experiment_id"] = experiment_id
        append_experiment_event(
            ledger_path,
            {
                "experiment_id": experiment_id,
                "status": "completed",
                "ended_at_utc": datetime.now(UTC).isoformat(),
                "train_corpus_sha256": result["train_rows_sha256"],
                "validation_corpus_sha256": result["validation_rows_sha256"],
                "config_sha256": result["training_config_sha256"],
                "best_step": result["best_checkpoint_step"],
                "validation_metrics": result["validation_best_metrics"],
            },
        )
        return result
    except BaseException as exc:
        append_experiment_event(
            ledger_path,
            {
                "experiment_id": experiment_id,
                "status": "failed",
                "ended_at_utc": datetime.now(UTC).isoformat(),
                "failure": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
