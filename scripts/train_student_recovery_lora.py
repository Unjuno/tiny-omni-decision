from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def emit_progress(path: Path, *, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    atomic_json(path, payload)
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _resolve(value: str | Path, *, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one option-level Recovery LoRA on a frozen packed ternary student."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--teacher-repo-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        import torch
        import yaml
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoProcessor

        from tiny_omni_decision.recovery import (
            select_decoder_recovery_targets,
            validate_recovery_config,
        )
        from tiny_omni_decision.schema import BaseModelManifest, DecisionExample
        from tiny_omni_decision.student import (
            model_sentence_embeddings,
            processor_inputs_for_decision_example,
            processor_inputs_for_options,
            supplied_option_logits,
        )
        from tiny_omni_decision.student_eval import (
            compare_student_predictions,
            evaluate_student_examples,
            load_fixed_validation_snapshot,
            validate_local_media_paths,
        )
        from tiny_omni_decision.student_evaluate import _verify_local_model_files
        from tiny_omni_decision.student_training import (
            load_teacher_option_cache,
            student_option_distillation_loss,
        )
        from tiny_omni_decision.ternary import load_packed_ternary_overlay
        from tiny_omni_decision.training import deterministic_sample_order
    except Exception as exc:
        raise RuntimeError(
            "Recovery requires the existing PyTorch/Transformers/PEFT CUDA environment"
        ) from exc

    repo = args.repo_root.resolve()
    teacher_repo = args.teacher_repo_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_recovery_config(config)
    student_cfg = config["student"]
    data_cfg = config["data"]
    teacher_cfg = config["teacher"]
    recovery_cfg = config["recovery"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]

    if not torch.cuda.is_available():
        raise RuntimeError("Recovery training requires the existing CUDA GPU")
    if int(train_cfg["seed"]) != 17:
        raise ValueError("the primary Recovery run is fixed to seed 17")
    if loss_cfg != {"option_kl": 1.0, "cross_entropy": 0.2, "brier": 0.2}:
        raise ValueError("Recovery loss must retain the frozen option-KL/CE/Brier weights")
    if not math.isclose(float(teacher_cfg["temperature"]), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the frozen Teacher temperature is 1.0")
    if not math.isclose(float(train_cfg["warmup_ratio"]), 0.03, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Recovery warmup ratio is frozen at 0.03")
    if train_cfg["lr_scheduler"] != "cosine":
        raise ValueError("Recovery scheduler is frozen to cosine")

    output_dir = _resolve(train_cfg["output_dir"], repo=repo)
    source_run_dir = Path(student_cfg["source_qat_run"]).resolve()
    overlay_dir = Path(student_cfg["ternary_overlay"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Recovery output: {output_dir}")
    if output_dir == source_run_dir or source_run_dir in output_dir.parents:
        raise ValueError("Recovery output must be separate from the source QAT run")
    source_metadata_path = source_run_dir / "run-metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("state") != "complete":
        raise ValueError("source QAT run is not complete")
    if sha256(source_metadata_path) != student_cfg["source_qat_run_metadata_sha256"]:
        raise ValueError("source QAT run metadata hash mismatch")
    if source_metadata.get("best_validation_step") != student_cfg["source_best_step"]:
        raise ValueError("configured QAT recovery source is not the selected best step")
    best_shadow_path = source_run_dir / "best-shadow.safetensors"
    if sha256(best_shadow_path) != student_cfg["source_best_shadow_sha256"]:
        raise ValueError("selected QAT shadow hash mismatch")
    if source_metadata.get("best_shadow_sha256") != student_cfg["source_best_shadow_sha256"]:
        raise ValueError("QAT ledger best shadow hash mismatch")
    overlay_manifest_path = overlay_dir / "manifest.json"
    overlay_tensor_path = overlay_dir / "weights.safetensors"
    if sha256(overlay_manifest_path) != student_cfg["ternary_overlay_manifest_sha256"]:
        raise ValueError("selected ternary overlay manifest hash mismatch")
    if sha256(overlay_tensor_path) != student_cfg["ternary_overlay_tensor_sha256"]:
        raise ValueError("selected ternary overlay tensor hash mismatch")
    overlay_manifest = json.loads(overlay_manifest_path.read_text(encoding="utf-8"))

    student_manifest_path = _resolve(student_cfg["model_manifest"], repo=repo)
    manifest = BaseModelManifest.model_validate(
        yaml.safe_load(student_manifest_path.read_text(encoding="utf-8"))
    )
    model_path = Path(student_cfg["model_path"]).resolve()
    _verify_local_model_files(model_path, manifest)
    if (
        overlay_manifest.get("base_model_id") != manifest.repo_id
        or overlay_manifest.get("base_revision") != manifest.revision
    ):
        raise ValueError("selected ternary overlay does not match the pinned student base")
    if source_metadata.get("student_revision") != manifest.revision:
        raise ValueError("source QAT run and student manifest revisions differ")

    train_path = _resolve(data_cfg["train_corpus"], repo=teacher_repo)
    data_root = _resolve(data_cfg["data_root"], repo=teacher_repo)
    validation_snapshot = Path(data_cfg["validation_snapshot"]).resolve()
    validation_manifest = _resolve(data_cfg["validation_selection_manifest"], repo=repo)
    source_validation_path = _resolve(data_cfg["source_validation_corpus"], repo=teacher_repo)
    teacher_validation_path = _resolve(
        data_cfg["teacher_validation_predictions"], repo=teacher_repo
    )
    cache_path = Path(teacher_cfg["train_cache"]).resolve()
    cache_metadata_path = Path(teacher_cfg["train_cache_metadata"]).resolve()
    required_files = (
        train_path,
        validation_snapshot,
        validation_manifest,
        source_validation_path,
        teacher_validation_path,
        cache_path,
        cache_metadata_path,
    )
    if any(not path.is_file() for path in required_files):
        missing = next(path for path in required_files if not path.is_file())
        raise FileNotFoundError(missing)
    if sha256(train_path) != data_cfg["train_corpus_sha256"]:
        raise ValueError("frozen training corpus hash mismatch")
    if sha256(validation_snapshot) != data_cfg["validation_snapshot_sha256"]:
        raise ValueError("frozen validation snapshot hash mismatch")
    if sha256(validation_manifest) != data_cfg["validation_selection_manifest_sha256"]:
        raise ValueError("frozen validation selection manifest hash mismatch")
    if sha256(cache_path) != teacher_cfg["train_cache_sha256"]:
        raise ValueError("frozen train-only Teacher cache hash mismatch")
    if sha256(cache_metadata_path) != teacher_cfg["train_cache_metadata_sha256"]:
        raise ValueError("frozen Teacher cache metadata hash mismatch")
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    if (
        cache_metadata.get("state") != "complete"
        or cache_metadata.get("identity", {}).get("teacher_id") != teacher_cfg["id"]
        or cache_metadata.get("identity", {}).get("teacher_revision") != teacher_cfg["revision"]
        or cache_metadata.get("identity", {}).get("temperature") != teacher_cfg["temperature"]
        or cache_metadata.get("identity", {}).get("train_corpus_sha256")
        != data_cfg["train_corpus_sha256"]
    ):
        raise ValueError("train-only Teacher cache identity differs from frozen config")

    train_examples = [
        DecisionExample.model_validate_json(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(example.split != "train" for example in train_examples):
        raise ValueError("Recovery training corpus contains non-train records")
    if len(train_examples) != int(train_cfg["max_train_examples"]):
        raise ValueError("Recovery training budget differs from the frozen corpus count")
    validation_examples, validation_teacher = load_fixed_validation_snapshot(
        validation_snapshot,
        validation_manifest,
        source_validation_path=source_validation_path,
        teacher_predictions_path=teacher_validation_path,
    )
    train_ids = {example.id for example in train_examples}
    validation_ids = {example.id for example in validation_examples}
    if train_ids & validation_ids:
        raise ValueError("train and validation IDs overlap")
    validate_local_media_paths(train_examples, data_root)
    validate_local_media_paths(validation_examples, data_root)
    cache = load_teacher_option_cache(
        cache_path,
        train_examples,
        expected_teacher_id=teacher_cfg["id"],
        expected_teacher_revision=teacher_cfg["revision"],
        expected_temperature=float(teacher_cfg["temperature"]),
    )
    if len(cache) != len(train_examples):
        raise ValueError("Teacher cache does not cover the full Recovery train corpus")

    accumulation = int(train_cfg["gradient_accumulation_steps"])
    total_steps = math.ceil(len(train_examples) / accumulation)
    if total_steps != int(train_cfg["max_steps"]):
        raise ValueError("configured Recovery steps differ from the one-pass train plan")
    order, sampling_counts = deterministic_sample_order(
        train_examples,
        seed=int(train_cfg["seed"]),
        limit=len(train_examples),
        modality_weights=train_cfg["modality_weights"],
        source_weights=train_cfg["source_weights"],
        max_sample_repeats=1,
    )
    ordered_ids = [example.id for example in order]
    if len(ordered_ids) != len(train_examples) or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("Recovery sampler must consume each unique train example once")
    train_order_sha256 = hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest()

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "adapter").mkdir()
    (output_dir / "effective-config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    history_path = output_dir / "training-history.jsonl"
    progress_path = output_dir / "latest-progress.json"
    started_at = datetime.now(UTC).isoformat()

    random.seed(int(train_cfg["seed"]))
    np.random.seed(int(train_cfg["seed"]))
    torch.manual_seed(int(train_cfg["seed"]))
    torch.cuda.manual_seed_all(int(train_cfg["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    base = AutoModel.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
    ).to("cuda")
    overlay_load = load_packed_ternary_overlay(
        base,
        overlay_dir,
        expected_base_model_id=manifest.repo_id,
        expected_base_revision=manifest.revision,
    )
    if overlay_load.get("tensor_file_sha256") != student_cfg["ternary_overlay_tensor_sha256"]:
        raise ValueError("loaded ternary overlay tensor hash changed")
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base_parameter_count = sum(parameter.numel() for parameter in base.parameters())
    linear_paths = [
        name for name, module in base.named_modules() if isinstance(module, torch.nn.Linear)
    ]
    target_modules = select_decoder_recovery_targets(linear_paths)
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=int(recovery_cfg["rank"]),
            lora_alpha=int(recovery_cfg["alpha"]),
            lora_dropout=float(recovery_cfg["dropout"]),
            bias="none",
            target_modules=list(target_modules),
            task_type=TaskType.FEATURE_EXTRACTION,
        ),
    )
    trainable_parameters = [
        parameter for parameter in peft_model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("PEFT did not create trainable Recovery parameters")
    if any(
        not name.endswith(("lora_A.default.weight", "lora_B.default.weight"))
        for name, p in peft_model.named_parameters()
        if p.requires_grad
    ):
        raise RuntimeError("only LoRA A/B parameters may be trainable")
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    if (
        sum(parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad)
        != trainable_parameter_count
    ):
        raise RuntimeError("unexpected trainable parameter inventory")
    peft_model.print_trainable_parameters()
    peft_model.gradient_checkpointing_disable() if hasattr(
        peft_model, "gradient_checkpointing_disable"
    ) else None

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(train_cfg["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=0.0,
        fused=True,
    )
    warmup_steps = max(1, round(total_steps * float(train_cfg["warmup_ratio"])))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    selector = config["training"]["best_checkpoint_selector"]
    best_key: tuple[float, float, float] | None = None
    best_step = -1
    best_adapter_dir = output_dir / "adapter" / "best"
    best_metrics_path: Path | None = None
    evaluation_seconds = 0.0
    training_started = time.perf_counter()
    rolling_losses: list[float] = []
    (output_dir / "sampling-plan.json").write_text(
        json.dumps(
            {
                "seed": int(train_cfg["seed"]),
                "sample_id_order_sha256": train_order_sha256,
                "unique_examples": len(set(ordered_ids)),
                "source_modality_counts": sampling_counts,
                "samples_per_optimizer_update": accumulation,
                "final_microbatch_examples": len(order) % accumulation or accumulation,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    def save_adapter(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)
        peft_model.save_pretrained(path, safe_serialization=True)

    def evaluate(step: int, examples_seen: int, mean_train_loss: float | None) -> None:
        nonlocal best_key, best_step, best_metrics_path, evaluation_seconds
        eval_started = time.perf_counter()
        emit_progress(
            progress_path,
            event="validation_start",
            step=step,
            examples_seen=examples_seen,
            validation_examples=len(validation_examples),
        )
        metrics, predictions = evaluate_student_examples(
            peft_model,
            processor,
            validation_examples,
            data_root=data_root,
            temperature=float(student_cfg["score_temperature"]),
            ece_bins=15,
        )
        paired = compare_student_predictions(predictions, validation_teacher, ece_bins=15)
        macro = metrics["macro_modality"]
        key = (
            float(macro["nll"]),
            -float(metrics["minimum_modality_accuracy"]),
            float(macro["brier"]),
        )
        eval_seconds = time.perf_counter() - eval_started
        evaluation_seconds += eval_seconds
        predictions_path = output_dir / f"validation-step-{step:04d}.jsonl"
        metrics_path = output_dir / f"validation-step-{step:04d}.json"
        write_jsonl(predictions_path, predictions)
        record = {
            "step": step,
            "examples_seen": examples_seen,
            "unique_examples_seen": examples_seen,
            "rolling_train_loss_mean": mean_train_loss,
            "validation_metrics": metrics,
            "validation_vs_frozen_teacher": paired,
            "selector_key": list(key),
            "best_step_before_evaluation": best_step,
            "is_best": best_key is None or key < best_key,
            "evaluation_seconds": eval_seconds,
            "predictions_sha256": sha256(predictions_path),
            "selector": selector,
        }
        if record["is_best"]:
            checkpoint_dir = output_dir / "checkpoints" / f"adapter-step-{step:04d}"
            if checkpoint_dir.exists():
                raise FileExistsError(checkpoint_dir)
            save_adapter(checkpoint_dir)
            if best_adapter_dir.exists():
                import shutil

                shutil.rmtree(best_adapter_dir)
            best_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copytree(checkpoint_dir, best_adapter_dir)
            best_key = key
            best_step = step
            record["best_step"] = step
        else:
            record["best_step"] = best_step
        atomic_json(metrics_path, record)
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        best_metrics_path = metrics_path if record["is_best"] else best_metrics_path
        emit_progress(
            progress_path,
            event="validation_done",
            step=step,
            examples_seen=examples_seen,
            validation_seconds=eval_seconds,
            best_step=best_step,
            is_best=record["is_best"],
        )
        peft_model.train()

    metadata: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "state": "running",
        "started_at_utc": started_at,
        "source_commit": __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
        .strip(),
        "config_sha256": sha256(config_path),
        "effective_config_path": str(output_dir / "effective-config.yaml"),
        "recovery_script_sha256": sha256(Path(__file__).resolve()),
        "student_code_sha256": {
            name: sha256(repo / "src" / "tiny_omni_decision" / name)
            for name in (
                "student.py",
                "student_eval.py",
                "student_training.py",
                "ternary.py",
                "recovery.py",
            )
        },
        "base_model_id": manifest.repo_id,
        "base_revision": manifest.revision,
        "base_manifest_sha256": sha256(student_manifest_path),
        "source_qat_run_dir": str(source_run_dir),
        "source_qat_run_metadata_sha256": sha256(source_metadata_path),
        "source_qat_best_step": int(student_cfg["source_best_step"]),
        "source_qat_best_shadow_sha256": sha256(best_shadow_path),
        "ternary_overlay_dir": str(overlay_dir),
        "ternary_overlay_manifest_sha256": sha256(overlay_manifest_path),
        "ternary_overlay_tensor_sha256": sha256(overlay_tensor_path),
        "train_corpus_sha256": sha256(train_path),
        "train_examples": len(train_examples),
        "train_order_sha256": train_order_sha256,
        "train_sampling_counts": sampling_counts,
        "teacher_cache_sha256": sha256(cache_path),
        "teacher_cache_metadata_sha256": sha256(cache_metadata_path),
        "teacher_id": teacher_cfg["id"],
        "teacher_revision": teacher_cfg["revision"],
        "teacher_temperature": float(teacher_cfg["temperature"]),
        "validation_snapshot_sha256": sha256(validation_snapshot),
        "validation_selection_manifest_sha256": sha256(validation_manifest),
        "validation_examples": len(validation_examples),
        "validation_ids_ordered_sha256": hashlib.sha256(
            "\n".join(example.id for example in validation_examples).encode("utf-8")
        ).hexdigest(),
        "sealed_audit_loaded": False,
        "loss": loss_cfg,
        "seed": int(train_cfg["seed"]),
        "planned_steps": total_steps,
        "gradient_accumulation_steps": accumulation,
        "optimizer": "AdamW",
        "learning_rate": float(train_cfg["learning_rate"]),
        "lr_scheduler": "cosine",
        "warmup_ratio": float(train_cfg["warmup_ratio"]),
        "warmup_steps": warmup_steps,
        "best_checkpoint_selector": selector,
        "recovery_rank": int(recovery_cfg["rank"]),
        "recovery_alpha": int(recovery_cfg["alpha"]),
        "recovery_dropout": float(recovery_cfg["dropout"]),
        "recovery_target_policy": recovery_cfg["target_policy"],
        "recovery_target_modules": list(target_modules),
        "base_parameter_count": base_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_base_parameter_count": base_parameter_count,
        "trainable_base_parameter_count": 0,
        "device": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "gpu_free_bytes_at_start": torch.cuda.mem_get_info(0)[0],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "transformers_version": importlib.metadata.version("transformers"),
        "peft_version": importlib.metadata.version("peft"),
        "peak_allocated_vram_bytes": 0,
        "cloud_compute_usd": 0,
        "product_teacher_promotion": False,
    }
    atomic_json(output_dir / "run-metadata.json", metadata)

    try:
        # LoRA initializes B to zero. Verify that wrapping the selected ternary base
        # preserves the saved QAT-best probabilities before any optimizer update.
        peft_model.eval()
        initial_metrics, initial_predictions = evaluate_student_examples(
            peft_model,
            processor,
            validation_examples,
            data_root=data_root,
            temperature=float(student_cfg["score_temperature"]),
            ece_bins=15,
        )
        qat_best_predictions_path = source_run_dir / "best-reload-predictions.jsonl"
        qat_best_predictions = [
            json.loads(line)
            for line in qat_best_predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(qat_best_predictions) != len(initial_predictions):
            raise ValueError("Recovery initial check has a different validation count")
        initial_differences: list[float] = []
        for expected, actual in zip(qat_best_predictions, initial_predictions, strict=True):
            if any(
                expected[key] != actual[key]
                for key in ("sample_id", "target", "options", "option_order_sha256")
            ):
                raise ValueError("Recovery initial check changed the validation pairing")
            initial_differences.extend(
                abs(float(left) - float(right))
                for left, right in zip(
                    expected["option_probabilities"],
                    actual["option_probabilities"],
                    strict=True,
                )
            )
        max_initial_difference = max(initial_differences)
        if max_initial_difference > 1e-5:
            raise ValueError(
                f"zero-init LoRA did not preserve QAT best ({max_initial_difference:.8g})"
            )
        write_jsonl(output_dir / "initial-qat-best-predictions.jsonl", initial_predictions)
        atomic_json(
            output_dir / "initial-qat-best-metrics.json",
            {
                "validation_metrics": initial_metrics,
                "validation_vs_frozen_teacher": compare_student_predictions(
                    initial_predictions, validation_teacher, ece_bins=15
                ),
                "max_qat_recovery_initial_probability_abs_difference": max_initial_difference,
                "predictions_sha256": sha256(output_dir / "initial-qat-best-predictions.jsonl"),
            },
        )
        save_adapter(best_adapter_dir)
        best_key = (
            float(initial_metrics["macro_modality"]["nll"]),
            -float(initial_metrics["minimum_modality_accuracy"]),
            float(initial_metrics["macro_modality"]["brier"]),
        )
        best_step = 0
        initial_checkpoint = output_dir / "checkpoints" / "adapter-step-0000"
        save_adapter(initial_checkpoint)
        best_metrics_path = output_dir / "initial-qat-best-metrics.json"
        peft_model.train()

        consumed = 0
        total_microbatches = 0
        interval = int(train_cfg["evaluation_interval"])
        max_grad_norm = float(train_cfg["max_gradient_norm"])
        for update_index, start in enumerate(range(0, len(order), accumulation), start=1):
            microbatch = order[start : start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            interval_losses: list[float] = []
            for example in microbatch:
                query_inputs = processor_inputs_for_decision_example(
                    processor, example, data_root=data_root
                )
                option_inputs = processor_inputs_for_options(processor, example.options)
                if max(
                    query_inputs["input_ids"].shape[-1],
                    option_inputs["input_ids"].shape[-1],
                ) > int(train_cfg["max_sequence_length"]):
                    raise ValueError(f"{example.id}: sequence exceeds configured maximum")
                query_embedding = model_sentence_embeddings(peft_model, query_inputs)[0]
                option_embeddings = model_sentence_embeddings(peft_model, option_inputs)
                logits = supplied_option_logits(
                    query_embedding,
                    option_embeddings,
                    temperature=float(student_cfg["score_temperature"]),
                )
                cached = cache[example.id]
                teacher_probabilities = torch.tensor(
                    cached["teacher_option_probabilities"],
                    device=logits.device,
                    dtype=torch.float32,
                )
                loss, _ = student_option_distillation_loss(
                    logits,
                    teacher_probabilities,
                    target_index=example.options.index(example.target),
                    option_kl_weight=float(loss_cfg["option_kl"]),
                    cross_entropy_weight=float(loss_cfg["cross_entropy"]),
                    brier_weight=float(loss_cfg["brier"]),
                )
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite Recovery loss for {example.id}")
                (loss / len(microbatch)).backward()
                interval_losses.append(float(loss.detach().cpu()))
                total_microbatches += 1
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_grad_norm,
                error_if_nonfinite=True,
                foreach=False,
            )
            if not torch.isfinite(gradient_norm):
                raise ValueError(f"non-finite Recovery gradient norm at step {update_index}")
            current_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            consumed += len(microbatch)
            rolling_losses.extend(interval_losses)
            if update_index == 1 or update_index % 4 == 0 or update_index == total_steps:
                emit_progress(
                    progress_path,
                    event="training_update",
                    step=update_index,
                    examples_seen=consumed,
                    total_steps=total_steps,
                    microbatch_loss_mean=sum(interval_losses) / len(interval_losses),
                    learning_rate_used=current_lr,
                    gradient_norm=float(gradient_norm.detach().cpu()),
                    allocated_vram_bytes=torch.cuda.memory_allocated(),
                    reserved_vram_bytes=torch.cuda.memory_reserved(),
                    elapsed_seconds=time.perf_counter() - training_started,
                )
            if update_index % interval == 0 or update_index == total_steps:
                evaluate(
                    update_index,
                    consumed,
                    sum(rolling_losses) / len(rolling_losses) if rolling_losses else None,
                )
                rolling_losses.clear()
            if (
                update_index % int(train_cfg["checkpoint_interval"]) == 0
                or update_index == total_steps
            ):
                checkpoint_dir = output_dir / "checkpoints" / f"latest-step-{update_index:04d}"
                save_adapter(checkpoint_dir)
                emit_progress(
                    progress_path,
                    event="checkpoint_saved",
                    step=update_index,
                    examples_seen=consumed,
                    adapter_dir=str(checkpoint_dir),
                )
            if update_index >= total_steps:
                break
        if consumed != len(train_examples) or total_microbatches != len(train_examples):
            raise RuntimeError("Recovery did not consume the full unique train corpus exactly once")
        if best_step < 0 or not best_adapter_dir.is_dir() or best_metrics_path is None:
            raise RuntimeError("Recovery did not produce a selected adapter checkpoint")

        final_adapter_dir = output_dir / "adapter" / "final"
        save_adapter(final_adapter_dir)
        final_adapter_weights = final_adapter_dir / "adapter_model.safetensors"
        best_adapter_weights = best_adapter_dir / "adapter_model.safetensors"
        best_reload_base = AutoModel.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
        ).to("cuda")
        loaded_overlay = load_packed_ternary_overlay(
            best_reload_base,
            overlay_dir,
            expected_base_model_id=manifest.repo_id,
            expected_base_revision=manifest.revision,
        )
        best_reload = (
            __import__("peft")
            .PeftModel.from_pretrained(
                best_reload_base,
                str(best_adapter_dir),
                is_trainable=False,
            )
            .eval()
        )
        reload_metrics, reload_predictions = evaluate_student_examples(
            best_reload,
            processor,
            validation_examples,
            data_root=data_root,
            temperature=float(student_cfg["score_temperature"]),
            ece_bins=15,
        )
        expected_path = output_dir / (
            "initial-qat-best-predictions.jsonl"
            if best_step == 0
            else f"validation-step-{best_step:04d}.jsonl"
        )
        expected_predictions = [
            json.loads(line)
            for line in expected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(expected_predictions) != len(reload_predictions):
            raise ValueError("best Recovery adapter reload changed validation row count")
        reload_differences: list[float] = []
        for expected, actual in zip(expected_predictions, reload_predictions, strict=True):
            if any(
                expected[key] != actual[key]
                for key in ("sample_id", "target", "options", "option_order_sha256")
            ):
                raise ValueError("best Recovery adapter reload changed validation pairing")
            reload_differences.extend(
                abs(float(left) - float(right))
                for left, right in zip(
                    expected["option_probabilities"],
                    actual["option_probabilities"],
                    strict=True,
                )
            )
        max_reload_difference = max(reload_differences)
        if max_reload_difference > 1e-5:
            raise ValueError(f"best Recovery adapter reload parity failed: {max_reload_difference}")
        write_jsonl(output_dir / "best-reload-predictions.jsonl", reload_predictions)
        reload_comparison = compare_student_predictions(
            reload_predictions, validation_teacher, ece_bins=15
        )
        atomic_json(
            output_dir / "best-reload-metrics.json",
            {
                "validation_metrics": reload_metrics,
                "validation_vs_frozen_teacher": reload_comparison,
                "paired_recovery_step": best_step,
                "max_in_training_reload_probability_abs_difference": max_reload_difference,
                "predictions_sha256": sha256(output_dir / "best-reload-predictions.jsonl"),
                "base_overlay_manifest_sha256": sha256(overlay_manifest_path),
                "base_overlay_tensor_sha256": loaded_overlay["tensor_file_sha256"],
            },
        )
        metadata.update(
            {
                "state": "complete",
                "ended_at_utc": datetime.now(UTC).isoformat(),
                "global_step": total_steps,
                "examples_consumed": consumed,
                "unique_examples_consumed": len(set(ordered_ids)),
                "training_microbatches": total_microbatches,
                "best_validation_step": best_step,
                "best_selector_key": list(best_key) if best_key is not None else None,
                "best_adapter_sha256": sha256(best_adapter_weights),
                "best_adapter_bytes": best_adapter_weights.stat().st_size,
                "final_adapter_sha256": sha256(final_adapter_weights),
                "final_adapter_bytes": final_adapter_weights.stat().st_size,
                "best_adapter_dir": str(best_adapter_dir),
                "final_adapter_dir": str(final_adapter_dir),
                "best_validation_metrics_sha256": sha256(best_metrics_path),
                "best_reload_metrics_sha256": sha256(output_dir / "best-reload-metrics.json"),
                "best_reload_max_probability_difference": max_reload_difference,
                "best_validation_metrics": reload_metrics,
                "validation_vs_frozen_teacher": reload_comparison,
                "evaluation_seconds_total": evaluation_seconds,
                "total_wall_seconds": time.perf_counter() - training_started,
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
                "sealed_audit_loaded": False,
                "product_teacher_promotion": False,
            }
        )
        atomic_json(output_dir / "run-metadata.json", metadata)
        emit_progress(
            progress_path,
            event="training_complete",
            step=total_steps,
            examples_seen=consumed,
            best_step=best_step,
            total_wall_seconds=metadata["total_wall_seconds"],
        )
    except Exception as exc:
        metadata["state"] = "failed"
        metadata["failure"] = f"{type(exc).__name__}: {exc}"
        metadata["failed_at_utc"] = datetime.now(UTC).isoformat()
        metadata["last_completed_step"] = (
            int(json.loads(progress_path.read_text(encoding="utf-8")).get("step", 0))
            if progress_path.exists()
            else 0
        )
        atomic_json(output_dir / "run-metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
