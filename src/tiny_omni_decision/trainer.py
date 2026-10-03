from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .corpus import comparison_deltas, file_sha256, selection_loss
from .decision import decision_loss, normalize_probabilities, option_logits_from_vocab
from .decision_math import brier_score, expected_calibration_error, negative_log_likelihood
from .io import load_structured_file
from .schema import BaseModelManifest, DecisionExample
from .training import (
    DecisionTrainingConfig,
    decision_training_config,
    deterministic_sample_order,
    output_record,
    processor_inputs_for_example,
    resolve_decoder_lora_targets,
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


def _forward_decision(
    model: Any,
    processor: Any,
    example: DecisionExample,
    *,
    data_root: Path,
    max_sequence_length: int,
) -> tuple[Any, int]:
    import torch

    inputs, option_ids, position, target = processor_inputs_for_example(
        processor, example, data_root=data_root
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


def run_training(
    *,
    train_path: Path,
    eval_path: Path,
    validation_path: Path | None = None,
    config_path: Path,
    output_dir: Path,
    seed_override: int | None = None,
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
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
    overrides = {
        "seed": seed_override,
        "max_train_examples": max_train_examples,
        "max_eval_examples": max_eval_examples,
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
    eval_raw = _read_examples(eval_path)
    validation_raw = _read_examples(validation_path) if validation_path else eval_raw
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
    if validation_path:
        for left_name, left, right_name, right in (
            ("train", train_raw, "validation", validation_raw),
            ("train", train_raw, "evaluation", eval_raw),
            ("validation", validation_raw, "evaluation", eval_raw),
        ):
            result = check_train_eval_splits(left, right)
            if result["status"] != "disjoint":
                raise ValueError(f"{left_name}/{right_name} contamination detected: {result}")

    train_ready, train_skipped = _eligible(train_raw, selected_modalities)
    eval_ready, eval_skipped = _eligible(eval_raw, selected_modalities)
    validation_ready, validation_skipped = _eligible(validation_raw, selected_modalities)
    if not train_ready or not eval_ready or not validation_ready:
        raise ValueError("no locally materialized training/validation/evaluation samples remain")
    train_order, _ = deterministic_sample_order(
        train_ready,
        seed=config.seed,
        limit=config.max_train_examples,
        modality_weights=config.modality_weights,
        source_weights=config.source_weights,
        max_sample_repeats=config.max_sample_repeats,
    )
    eval_subset, _ = deterministic_sample_order(
        eval_ready,
        seed=config.seed,
        limit=min(config.max_eval_examples, len(eval_ready)),
        modality_weights=config.modality_weights,
        source_weights=config.source_weights,
    )
    validation_subset, _ = deterministic_sample_order(
        validation_ready,
        seed=config.seed + 1,
        limit=min(config.selection_eval_examples, len(validation_ready)),
        modality_weights=config.modality_weights,
        source_weights=config.source_weights,
    )
    if tiny_overfit:
        train_order = train_order[: min(len(train_order), 8)]
        eval_subset = train_order
        validation_subset = train_order
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
    base = AutoModelForMultimodalLM.from_pretrained(
        manifest.repo_id,
        revision=manifest.revision,
        dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    data_root = train_path.parent.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics, baseline_predictions = _evaluate(
        base, processor, eval_subset, data_root=data_root, config=config
    )
    validation_baseline_metrics, validation_baseline_predictions = _evaluate(
        base, processor, validation_subset, data_root=data_root, config=config
    )
    (output_dir / "baseline-evaluation.json").write_text(
        json.dumps(baseline_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "baseline-predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in baseline_predictions), encoding="utf-8"
    )
    (output_dir / "validation-baseline-evaluation.json").write_text(
        json.dumps(validation_baseline_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "validation-baseline-predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in validation_baseline_predictions),
        encoding="utf-8",
    )
    tiny_before = None
    if tiny_overfit:
        probabilities = [record["option_probabilities"] for record in baseline_predictions]
        targets = [record["target"] for record in baseline_predictions]
        tiny_before = config.cross_entropy_weight * negative_log_likelihood(
            probabilities, targets
        ) + config.brier_weight * brier_score(probabilities, targets)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    targets = resolve_decoder_lora_targets(base)
    if resume_from:
        model = PeftModel.from_pretrained(base, resume_from, is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    train_hash = file_sha256(str(train_path))
    eval_hash = file_sha256(str(eval_path))
    validation_hash = file_sha256(str(validation_path or eval_path))
    config_hash = file_sha256(str(config_path))
    if resume_from and (resume_from / "optimizer.pt").is_file():
        optimizer.load_state_dict(
            torch.load(resume_from / "optimizer.pt", map_location="cpu", weights_only=True)
        )
    model.train()
    consumed: Counter[str] = Counter()
    losses: list[float] = []
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    sample_index = 0
    global_step = 0
    best_selection_loss = float("inf")
    best_step = 0
    best_validation_predictions: list[dict[str, Any]] | None = None
    if resume_from and (resume_from / "trainer-state.json").is_file():
        saved_state = json.loads((resume_from / "trainer-state.json").read_text(encoding="utf-8"))
        expected_hashes = {
            "train_rows_sha256": train_hash,
            "eval_rows_sha256": eval_hash,
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
            "consumed": dict(consumed),
            "losses": losses,
            "history": history,
            "train_rows_sha256": train_hash,
            "eval_rows_sha256": eval_hash,
            "validation_rows_sha256": validation_hash,
            "config_sha256": config_hash,
            "base_revision": manifest.revision,
        }

    while global_step < config.max_steps:
        step_total_losses: list[float] = []
        step_cross_entropies: list[float] = []
        step_briers: list[float] = []
        step_composition: Counter[str] = Counter()
        for _ in range(config.gradient_accumulation_steps):
            example = train_order[sample_index % len(train_order)]
            sample_index += 1
            batch_key = f"{example.modality}:{example.source}"
            consumed[batch_key] += 1
            step_composition[batch_key] += 1
            logits, target = _forward_decision(
                model,
                processor,
                example,
                data_root=data_root,
                max_sequence_length=config.max_sequence_length,
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
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if [parameter._version for parameter in frozen_parameters] != base_versions:
            raise RuntimeError("a frozen base parameter changed during training")
        step_record = {
            "step": global_step,
            "cross_entropy": sum(step_cross_entropies) / len(step_cross_entropies),
            "brier": sum(step_briers) / len(step_briers),
            "total_loss": sum(step_total_losses) / len(step_total_losses),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": gradient_norm,
            "sample_composition": dict(sorted(step_composition.items())),
        }
        history.append(step_record)
        with history_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(step_record) + "\n")

        selection_metrics = None
        if global_step % config.evaluation_interval == 0 or global_step == config.max_steps:
            selection_metrics, selection_predictions = _evaluate(
                model, processor, validation_subset, data_root=data_root, config=config
            )
            score = selection_loss(selection_metrics["all"], brier_weight=config.brier_weight)
            selection_metrics["selection_loss"] = score
            (output_dir / f"validation-step-{global_step}.json").write_text(
                json.dumps(selection_metrics, indent=2), encoding="utf-8"
            )
            if score < best_selection_loss:
                best_selection_loss = score
                best_step = global_step
                best_validation_predictions = selection_predictions
                temporary_best = output_dir / "best.tmp"
                if temporary_best.exists():
                    shutil.rmtree(temporary_best)
                model.save_pretrained(temporary_best)
                if best_path.exists():
                    shutil.rmtree(best_path)
                temporary_best.replace(best_path)
                (output_dir / "best-checkpoint.json").write_text(
                    json.dumps(
                        {"step": best_step, "selection_loss": best_selection_loss}, indent=2
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
            (checkpoint_path / "trainer-state.json").write_text(
                json.dumps(state_record(global_step)), encoding="utf-8"
            )
        if global_step % config.evaluation_interval == 0 or global_step == config.max_steps:
            (output_dir / "trainer-state.json").write_text(
                json.dumps(state_record(global_step)), encoding="utf-8"
            )
        model.train()
    if best_step == 0:
        raise RuntimeError("training ended without selecting a best validation checkpoint")

    # Load the chosen validation checkpoint from disk before acceptance evaluation.
    model.eval()
    unloaded_base = model.unload()
    del model
    gc.collect()
    reloaded = PeftModel.from_pretrained(unloaded_base, best_path, is_trainable=False).eval()
    reloaded_validation_metrics, reloaded_validation_predictions = _evaluate(
        reloaded, processor, validation_subset, data_root=data_root, config=config
    )
    if best_validation_predictions is None or [
        record["option_probabilities"] for record in best_validation_predictions
    ] != [record["option_probabilities"] for record in reloaded_validation_predictions]:
        raise RuntimeError("best checkpoint save/reload validation predictions changed")
    reloaded_metrics, reloaded_predictions = _evaluate(
        reloaded, processor, eval_subset, data_root=data_root, config=config
    )
    tiny_after = None
    if tiny_overfit:
        probabilities = [record["option_probabilities"] for record in reloaded_predictions]
        targets = [record["target"] for record in reloaded_predictions]
        tiny_after = config.cross_entropy_weight * negative_log_likelihood(
            probabilities, targets
        ) + config.brier_weight * brier_score(probabilities, targets)
        if tiny_after >= tiny_before:
            raise RuntimeError(f"tiny-overfit loss did not decrease: {tiny_before} -> {tiny_after}")

    torch.cuda.synchronize()
    max_vram = torch.cuda.max_memory_allocated()
    end_at = datetime.now(UTC)
    wall_seconds = time.monotonic() - started_clock
    source_counts = Counter(example.source for example in eval_subset)
    modality_counts = Counter(example.modality for example in eval_subset)
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
    metadata = {
        "teacher_id": "tiny-omni-decision-teacher-v0",
        "artifact_role": "high_precision_decision_teacher",
        "base_repo_id": manifest.repo_id,
        "base_revision": manifest.revision,
        "seed": config.seed,
        "config": config.model_dump(mode="json"),
        "target_modules": targets,
        "base_weights_sha256": next(
            (item["sha256"] for item in manifest.files if item.get("path") == "model.safetensors"),
            None,
        ),
        "train_rows_sha256": train_hash,
        "eval_rows_sha256": eval_hash,
        "validation_rows_sha256": validation_hash,
        "corpus_manifest_sha256": file_sha256(
            str(train_path.parent / "corpus-manifest.json")
        ),
        "training_config_sha256": config_hash,
        "train_examples_available": len(train_ready),
        "train_examples_selected": len(train_order),
        "validation_examples_available": len(validation_ready),
        "validation_examples_measured": len(validation_subset),
        "eval_examples_measured": len(eval_subset),
        "skipped_train_examples": train_skipped,
        "skipped_validation_examples": validation_skipped,
        "skipped_eval_examples": eval_skipped,
        "actual_train_consumption_by_modality_source": dict(sorted(consumed.items())),
        "actual_train_consumption_by_modality": dict(sorted(modality_consumption.items())),
        "train_loss_mean": sum(losses) / len(losses),
        "tiny_overfit_nll_before": tiny_before,
        "tiny_overfit_nll_after": tiny_after,
        "evaluation_source_counts": dict(sorted(source_counts.items())),
        "evaluation_modality_counts": dict(sorted(modality_counts.items())),
        "baseline_metrics": baseline_metrics,
        "decision_teacher_metrics": reloaded_metrics,
        "metric_deltas_teacher_minus_base": comparison_deltas(
            baseline_metrics, reloaded_metrics
        ),
        "validation_baseline_metrics": validation_baseline_metrics,
        "validation_best_metrics": reloaded_validation_metrics,
        "validation_best_selection_loss": best_selection_loss,
        "best_checkpoint_step": best_step,
        "best_adapter_path": "best",
        "step_checkpoint_sha256": checkpoint_hashes,
        "best_adapter_sha256": file_sha256(str(best_weights_path)),
        "adapter_config_sha256": file_sha256(str(best_path / "adapter_config.json")),
        "global_steps": global_step,
        "optimizer_microbatches": sample_index,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "max_gradient_norm": config.max_gradient_norm,
        "sampling_mixture": {
            "modality_weights": config.modality_weights,
            "source_weights": config.source_weights,
            "max_sample_repeats": config.max_sample_repeats,
            "scheduler": (
                "weighted modality round robin; source weights apply within modality; "
                "sample reuse is bounded by max_sample_repeats"
            ),
            "consumed_by_modality_source": dict(sorted(consumed.items())),
        },
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "max_allocated_vram_bytes": max_vram,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
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
    (output_dir / "decision-lora-predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in reloaded_predictions), encoding="utf-8"
    )
    (output_dir / "decision-teacher-evaluation.json").write_text(
        json.dumps(reloaded_metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "validation-teacher-evaluation.json").write_text(
        json.dumps(reloaded_validation_metrics, indent=2), encoding="utf-8"
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
    corpus_manifest_path = train_path.parent / "corpus-manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    teacher_manifest = {
        "schema_version": 1,
        "teacher_id": "tiny-omni-decision-teacher-v0",
        "artifact_role": "high_precision_reference_before_ternary_compression",
        "adapter_path": "best",
        "adapter_weights_file": best_weights_path.name,
        "adapter_weights_sha256": metadata["best_adapter_sha256"],
        "adapter_config_sha256": metadata["adapter_config_sha256"],
        "base_model": {
            "repo_id": manifest.repo_id,
            "revision": manifest.revision,
            "model_manifest_sha256": file_sha256(str(model_manifest_path)),
            "weights_sha256": metadata["base_weights_sha256"],
        },
        "corpus": {
            "manifest_path": str(corpus_manifest_path),
            "manifest_sha256": metadata["corpus_manifest_sha256"],
            "train_jsonl_sha256": train_hash,
            "validation_jsonl_sha256": validation_hash,
            "evaluation_jsonl_sha256": eval_hash,
            "pair_sha256": corpus_manifest.get("corpus_pair_sha256"),
            "overlap": corpus_manifest.get("overlap"),
            "counts": {
                "train": corpus_manifest.get("train"),
                "validation": corpus_manifest.get("validation"),
                "evaluation": corpus_manifest.get("evaluation"),
            },
        },
        "training_config": config.model_dump(mode="json"),
        "training_config_sha256": config_hash,
        "seed": config.seed,
        "target_module_paths": targets,
        "sampling_mixture": metadata["sampling_mixture"],
        "optimizer_steps": global_step,
        "best_checkpoint_step": best_step,
        "best_checkpoint_selection_loss": best_selection_loss,
        "checkpoint_sha256": checkpoint_hashes,
        "run_metadata_path": "run-metadata.json",
        "run_metadata_sha256": file_sha256(str(output_dir / "run-metadata.json")),
        "baseline_metrics_path": "baseline-evaluation.json",
        "baseline_predictions_path": "baseline-predictions.jsonl",
        "teacher_metrics_path": "decision-teacher-evaluation.json",
        "teacher_predictions_path": "decision-lora-predictions.jsonl",
        "metrics": {
            "base": baseline_metrics,
            "teacher": reloaded_metrics,
            "delta_teacher_minus_base": metadata["metric_deltas_teacher_minus_base"],
        },
        "environment": {
            key: metadata[key]
            for key in (
                "python_version",
                "torch_version",
                "cuda_runtime_version",
                "transformers_version",
                "peft_version",
                "accelerate_version",
                "gpu",
                "gpu_total_memory_bytes",
            )
        },
        "training_wall_seconds": wall_seconds,
        "rental_provider": "local workstation",
        "rental_total_usd": 0.0,
        "merge_export_performed": False,
        "ternary_quantization_performed": False,
        "recovery_lora_trained": False,
    }
    (output_dir / "teacher-manifest.json").write_text(
        json.dumps(teacher_manifest, indent=2), encoding="utf-8"
    )
    return metadata
