from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
    train_raw = _read_examples(train_path)
    eval_raw = _read_examples(eval_path)
    train_ready, train_skipped = _eligible(train_raw, selected_modalities)
    eval_ready, eval_skipped = _eligible(eval_raw, selected_modalities)
    if not train_ready or not eval_ready:
        raise ValueError("no locally materialized training/evaluation samples remain")
    train_order, _ = deterministic_sample_order(
        train_ready,
        seed=config.seed,
        limit=min(config.max_train_examples, len(train_ready)),
        modality_weights=config.modality_weights,
        source_weights=config.source_weights,
    )
    eval_subset, _ = deterministic_sample_order(
        eval_ready,
        seed=config.seed,
        limit=min(config.max_eval_examples, len(eval_ready)),
        modality_weights=config.modality_weights,
        source_weights=config.source_weights,
    )
    if tiny_overfit:
        train_order = train_order[: min(len(train_order), 8)]
        eval_subset = train_order

    data_root = train_path.parent.parent.parent
    baseline_metrics, baseline_predictions = _evaluate(
        base, processor, eval_subset, data_root=data_root, config=config
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
    if resume_from and (resume_from / "optimizer.pt").is_file():
        optimizer.load_state_dict(
            torch.load(resume_from / "optimizer.pt", map_location="cpu", weights_only=True)
        )
    model.train()
    consumed: Counter[str] = Counter()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    sample_index = 0
    global_step = 0
    if resume_from and (resume_from / "trainer-state.json").is_file():
        saved_state = json.loads((resume_from / "trainer-state.json").read_text(encoding="utf-8"))
        global_step = int(saved_state["global_step"])
        sample_index = int(saved_state.get("sample_index", 0))
    while global_step < config.max_steps:
        micro_losses = []
        for _ in range(config.gradient_accumulation_steps):
            example = train_order[sample_index % len(train_order)]
            sample_index += 1
            consumed[f"{example.modality}:{example.source}"] += 1
            logits, target = _forward_decision(
                model,
                processor,
                example,
                data_root=data_root,
                max_sequence_length=config.max_sequence_length,
            )
            loss, _, _ = decision_loss(
                logits.unsqueeze(0),
                torch.tensor([target], device=logits.device),
                brier_weight=config.brier_weight,
                cross_entropy_weight=config.cross_entropy_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at sample {example.id}")
            (loss / config.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach()))
            micro_losses.append(float(loss.detach()))
        for parameter in adapter_parameters:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Decision LoRA has a missing or non-finite gradient")
        if any(parameter.grad is not None for parameter in frozen_parameters):
            raise RuntimeError("frozen base received gradients")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if [parameter._version for parameter in frozen_parameters] != base_versions:
            raise RuntimeError("a frozen base parameter changed during training")
        if global_step % config.checkpoint_interval == 0 or global_step == config.max_steps:
            model.save_pretrained(output_dir)
            torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
            (output_dir / "trainer-state.json").write_text(
                json.dumps({"global_step": global_step, "sample_index": sample_index}),
                encoding="utf-8",
            )
        if global_step % config.evaluation_interval == 0 or global_step == config.max_steps:
            interval_examples = eval_subset[: min(4, len(eval_subset))]
            interval_metrics, _ = _evaluate(
                model, processor, interval_examples, data_root=data_root, config=config
            )
            (output_dir / f"evaluation-step-{global_step}.json").write_text(
                json.dumps(interval_metrics, indent=2), encoding="utf-8"
            )
        model.train()

    model.eval()
    tuned_metrics, tuned_predictions = _evaluate(
        model, processor, eval_subset, data_root=data_root, config=config
    )
    tiny_after = None
    if tiny_overfit:
        probabilities = [record["option_probabilities"] for record in tuned_predictions]
        targets = [record["target"] for record in tuned_predictions]
        tiny_after = config.cross_entropy_weight * negative_log_likelihood(
            probabilities, targets
        ) + config.brier_weight * brier_score(probabilities, targets)
        if tiny_after >= tiny_before:
            raise RuntimeError(f"tiny-overfit loss did not decrease: {tiny_before} -> {tiny_after}")

    # Preserve the high-precision adapter, then prove the same readout survives a reload.
    del tuned_metrics
    gc.collect()
    reloaded_base = base
    reloaded = PeftModel.from_pretrained(reloaded_base, output_dir, is_trainable=False).eval()
    with torch.no_grad():
        check_logits, _ = _forward_decision(
            reloaded,
            processor,
            eval_subset[0],
            data_root=data_root,
            max_sequence_length=config.max_sequence_length,
        )
        check_probabilities = normalize_probabilities(check_logits).cpu().tolist()
    if (
        not all(value == value for value in check_probabilities)
        or abs(sum(check_probabilities) - 1.0) > 1e-5
    ):
        raise RuntimeError("reloaded adapter probability check failed")
    reloaded_metrics, reloaded_predictions = _evaluate(
        reloaded, processor, eval_subset, data_root=data_root, config=config
    )
    if [record["option_probabilities"] for record in tuned_predictions] != [
        record["option_probabilities"] for record in reloaded_predictions
    ]:
        raise RuntimeError("adapter save/reload predictions changed")
    torch.cuda.synchronize()
    max_vram = torch.cuda.max_memory_allocated()
    source_counts = Counter(example.source for example in eval_subset)
    metadata = {
        "base_repo_id": manifest.repo_id,
        "base_revision": manifest.revision,
        "seed": config.seed,
        "config": config.model_dump(mode="json"),
        "target_modules": targets,
        "base_weights_sha256": next(
            (item["sha256"] for item in manifest.files if item.get("path") == "model.safetensors"),
            None,
        ),
        "train_rows_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "eval_rows_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        "corpus_manifest_sha256": hashlib.sha256(
            (train_path.parent / "corpus-manifest.json").read_bytes()
        ).hexdigest(),
        "training_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "train_examples_available": len(train_ready),
        "eval_examples_measured": len(eval_subset),
        "skipped_train_examples": train_skipped,
        "skipped_eval_examples": eval_skipped,
        "actual_train_consumption_by_modality_source": dict(sorted(consumed.items())),
        "train_loss_mean": sum(losses) / len(losses),
        "tiny_overfit_nll_before": tiny_before,
        "tiny_overfit_nll_after": tiny_after,
        "evaluation_source_counts": dict(sorted(source_counts.items())),
        "baseline_metrics": baseline_metrics,
        "decision_lora_metrics": reloaded_metrics,
        "baseline_predictions": baseline_predictions,
        "decision_lora_predictions": reloaded_predictions,
        "global_steps": global_step,
        "optimizer_microbatches": sample_index,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "max_allocated_vram_bytes": max_vram,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
        "accelerate_version": importlib.metadata.version("accelerate"),
        "python_version": __import__("platform").python_version(),
        "checkpoint_reload_verified": True,
    }
    (output_dir / "run-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "baseline-predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in baseline_predictions), encoding="utf-8"
    )
    (output_dir / "decision-lora-predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in reloaded_predictions), encoding="utf-8"
    )
    return metadata
