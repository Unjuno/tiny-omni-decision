from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
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


def emit_progress(path: Path, *, event: str, **fields: Any) -> None:
    """Publish an atomic, human-readable training heartbeat."""
    payload = {"event": event, **fields}
    atomic_json(path, payload)
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def verify_warm_start_ledger(
    previous: dict[str, Any],
    expected: dict[str, Any],
    *,
    total_steps: int,
    checkpoint_interval: int,
) -> int:
    """Validate an explicitly requested shadow-only, optimizer-reset handoff."""
    for field, value in expected.items():
        if previous.get(field) != value:
            raise ValueError(f"warm-start source differs in {field}")
    step = previous.get("best_validation_step")
    current = previous.get("global_step")
    if type(step) is not int or type(current) is not int or not (
        0 < step <= current <= total_steps
    ):
        raise ValueError("warm-start source has no completed selected checkpoint")
    if checkpoint_interval <= 0 or step % checkpoint_interval:
        raise ValueError("warm-start selected step is not on the checkpoint interval")
    digest = previous.get("best_shadow_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("warm-start source best-shadow digest is invalid")
    return step


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a ternary-constrained EmbeddingGemma student with frozen Teacher options."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--teacher-repo-root", type=Path, required=True)
    parser.add_argument(
        "--warm-start-run", type=Path,
        help=("Selected BF16 best-shadow from a stopped earlier run. "
              "Resets optimizer and scheduler state; NOT an exact resume."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import yaml
    from safetensors.torch import load_file, save_file
    from transformers import AutoModel, AutoProcessor

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from tiny_omni_decision.io import load_structured_file
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
        build_ternary_qat_optimizer,
        copy_shadow_gradients_to_masters,
        load_teacher_option_cache,
        make_fp32_cpu_master_parameters,
        restore_qat_shadows,
        student_option_distillation_loss,
        sync_cpu_masters_to_shadows,
    )
    from tiny_omni_decision.ternary import (
        apply_ternary_qat,
        cached_ternary_validation,
        export_packed_ternary_overlay,
        load_packed_ternary_overlay,
    )
    from tiny_omni_decision.training import deterministic_sample_order

    repo = args.repo_root.resolve()
    teacher_repo = args.teacher_repo_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported student QAT config")
    if not str(config.get("experiment_id", "")).startswith(
        "embeddinggemma2-ternary-qat-v0"
    ):
        raise ValueError("unexpected experiment ID")
    train_cfg = config["training"]
    teacher_cfg = config["teacher"]
    student_cfg = config["student"]
    loss_cfg = config["loss"]
    if train_cfg["device"] != "cuda" or train_cfg["dtype"] != "bfloat16":
        raise ValueError("this local experiment is frozen to CUDA BF16")
    if train_cfg["early_stopping"] is not False:
        raise ValueError("the configured full one-pass training budget may not early stop")
    if train_cfg["optimizer"].lower() != student_cfg["ternary"]["optimizer"].lower():
        raise ValueError("training and ternary optimizer declarations differ")
    parameter_device_policy = train_cfg.get("optimizer_parameter_device", "model")
    if parameter_device_policy not in {"model", "cpu_fp32_master"}:
        raise ValueError("unsupported QAT optimizer parameter device policy")
    if train_cfg["seed"] != 17 or train_cfg["max_sample_repeats"] != 1:
        raise ValueError("seed 17 and no repeated training examples are required")
    if config["loss"] != {"option_kl": 1.0, "cross_entropy": 0.2, "brier": 0.2}:
        raise ValueError("loss coefficients differ from the recorded fixed objective")
    if teacher_cfg["teacher_temperature"] != 1.0 or student_cfg["score_temperature"] != 0.1:
        raise ValueError("Teacher/student temperatures differ from the fixed experiment config")
    if not torch.cuda.is_available():
        raise RuntimeError("ternary QAT requires the existing local CUDA GPU")
    free_vram, total_vram = torch.cuda.mem_get_info()
    if free_vram < 7 * 1024**3:
        raise RuntimeError(f"insufficient free VRAM: {free_vram / 1024**3:.2f} GiB")

    def resolve(path_value: str, base: Path) -> Path:
        path = Path(path_value)
        return path.resolve() if path.is_absolute() else (base / path).resolve()

    student_manifest_path = resolve(student_cfg["model_manifest"], repo)
    model_path = Path(student_cfg["model_path"]).resolve()
    teacher_manifest_path = resolve(teacher_cfg["manifest"], teacher_repo)
    teacher_run_metadata_path = resolve(teacher_cfg["run_metadata"], teacher_repo)
    teacher_base_manifest_path = resolve(teacher_cfg["base_model_manifest"], teacher_repo)
    teacher_adapter_dir = resolve(teacher_cfg["adapter"], teacher_repo)
    train_path = resolve(teacher_cfg["train_corpus"], teacher_repo)
    data_root = teacher_repo / "data"
    cache_path = Path(teacher_cfg["train_cache"]).resolve()
    cache_metadata_path = cache_path.parent / "cache-run.json"
    validation_snapshot = resolve(teacher_cfg["validation_snapshot"], repo)
    validation_manifest = resolve(teacher_cfg["validation_manifest"], repo)
    validation_corpus = resolve(teacher_cfg["validation_corpus"], teacher_repo)
    validation_teacher_predictions = resolve(teacher_cfg["validation_predictions"], teacher_repo)
    initial_overlay_dir = Path(student_cfg["initial_ternary_overlay"]).resolve()
    initial_overlay_manifest_path = initial_overlay_dir / "manifest.json"
    initial_ternary_predictions_path = Path(
        student_cfg["initial_ternary_validation_predictions"]
    ).resolve()
    initial_metrics_path = initial_ternary_predictions_path.with_name("metrics.json")
    output_dir = Path(train_cfg["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing QAT run: {output_dir}")
    for path in (
        student_manifest_path,
        model_path / "model.safetensors",
        train_path,
        cache_path,
        cache_metadata_path,
        validation_snapshot,
        validation_manifest,
        validation_corpus,
        validation_teacher_predictions,
        initial_ternary_predictions_path,
        initial_metrics_path,
        initial_overlay_manifest_path,
        teacher_manifest_path,
        teacher_run_metadata_path,
        teacher_base_manifest_path,
        teacher_adapter_dir / "adapter_model.safetensors",
        teacher_adapter_dir / "adapter_config.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    teacher_run = json.loads(teacher_run_metadata_path.read_text(encoding="utf-8"))
    teacher_base_manifest = BaseModelManifest.model_validate(
        load_structured_file(teacher_base_manifest_path)
    )
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    if teacher_manifest.get("teacher_id") != teacher_cfg["teacher_id"]:
        raise ValueError("frozen Teacher manifest ID differs from config")
    if teacher_manifest.get("base_model", {}).get("revision") != teacher_cfg["teacher_revision"]:
        raise ValueError("frozen Teacher manifest revision differs from config")
    if teacher_run.get("teacher_id") != teacher_cfg["teacher_id"] or not teacher_run.get(
        "checkpoint_reload_verified"
    ):
        raise ValueError("frozen Teacher run does not verify the selected checkpoint reload")
    if (
        teacher_run.get("best_checkpoint_step")
        != teacher_manifest["artifact"]["best_checkpoint_step"]
    ):
        raise ValueError("frozen Teacher manifest and run metadata select different checkpoints")
    if (
        teacher_base_manifest.repo_id != teacher_manifest["base_model"]["repo_id"]
        or teacher_base_manifest.revision != teacher_manifest["base_model"]["revision"]
    ):
        raise ValueError("Teacher and base-model manifest identities differ")
    if (
        sha256(teacher_adapter_dir / "adapter_model.safetensors")
        != teacher_manifest["artifact"]["weights_sha256"]
    ):
        raise ValueError("frozen Teacher adapter hash does not match its manifest")
    if (
        sha256(teacher_adapter_dir / "adapter_config.json")
        != teacher_manifest["artifact"]["config_sha256"]
    ):
        raise ValueError("frozen Teacher adapter config hash does not match its manifest")
    if (
        sha256(teacher_base_manifest_path)
        != teacher_manifest["base_model"]["model_manifest_sha256"]
    ):
        raise ValueError("frozen Teacher base-model manifest hash mismatch")
    if cache_metadata.get("state") != "complete" or cache_metadata.get("cache_sha256") != sha256(
        cache_path
    ):
        raise ValueError("train-only Teacher cache is incomplete or does not match its metadata")
    if cache_metadata["identity"]["train_corpus_sha256"] != sha256(train_path):
        raise ValueError("Teacher cache was generated from a different training corpus")
    if cache_metadata["identity"]["temperature"] != teacher_cfg["teacher_temperature"]:
        raise ValueError("Teacher cache temperature differs from the frozen config")
    if cache_metadata["identity"]["teacher_manifest_sha256"] != sha256(teacher_manifest_path):
        raise ValueError("Teacher cache was generated from a different Teacher manifest")
    if cache_metadata["identity"]["teacher_run_metadata_sha256"] != sha256(
        teacher_run_metadata_path
    ):
        raise ValueError("Teacher cache was generated from different run metadata")
    if cache_metadata["identity"]["base_manifest_sha256"] != sha256(teacher_base_manifest_path):
        raise ValueError("Teacher cache was generated from a different base-model manifest")
    if cache_metadata["identity"]["teacher_id"] != teacher_cfg["teacher_id"]:
        raise ValueError("Teacher cache identity differs from the frozen config")
    if cache_metadata["identity"]["teacher_revision"] != teacher_cfg["teacher_revision"]:
        raise ValueError("Teacher cache base revision differs from the frozen config")

    manifest = BaseModelManifest.model_validate(load_structured_file(student_manifest_path))
    _verify_local_model_files(model_path, manifest)
    with train_path.open(encoding="utf-8") as stream:
        train_examples = [
            DecisionExample.model_validate_json(line) for line in stream if line.strip()
        ]
    if not train_examples or any(example.split != "train" for example in train_examples):
        raise ValueError("QAT input must be the frozen train split only")
    teacher_cache = load_teacher_option_cache(
        cache_path,
        train_examples,
        expected_teacher_id=teacher_cfg["teacher_id"],
        expected_teacher_revision=teacher_cfg["teacher_revision"],
        expected_temperature=teacher_cfg["teacher_temperature"],
    )
    validation_examples, validation_teacher = load_fixed_validation_snapshot(
        validation_snapshot,
        validation_manifest,
        source_validation_path=validation_corpus,
        teacher_predictions_path=validation_teacher_predictions,
    )
    validate_local_media_paths(validation_examples, data_root)
    train_ids = {example.id for example in train_examples}
    validation_ids = {example.id for example in validation_examples}
    if train_ids & validation_ids:
        raise ValueError("train and validation sample IDs overlap")
    if len(train_examples) != int(train_cfg["max_train_examples"]):
        raise ValueError("configured full train budget does not match frozen corpus rows")
    planned_steps = math.ceil(len(train_examples) / int(train_cfg["gradient_accumulation_steps"]))
    if planned_steps != int(train_cfg["max_steps"]):
        raise ValueError(
            f"configured steps {train_cfg['max_steps']} != one-pass plan {planned_steps}"
        )
    order, planned_counts = deterministic_sample_order(
        train_examples,
        seed=int(train_cfg["seed"]),
        limit=len(train_examples),
        modality_weights=train_cfg["modality_weights"],
        source_weights=train_cfg["source_weights"],
        max_sample_repeats=1,
    )
    ordered_ids = [example.id for example in order]
    if len(ordered_ids) != len(train_examples) or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("sampler did not produce one pass over every unique train example")
    order_hash = hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest()
    initial_predictions = [
        json.loads(line)
        for line in initial_ternary_predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_initial_hash = student_cfg["initial_ternary_validation_predictions_sha256"]
    if (
        sha256(initial_overlay_manifest_path)
        != student_cfg["initial_ternary_overlay_manifest_sha256"]
    ):
        raise ValueError("initial ternary overlay manifest hash changed")
    initial_overlay_manifest = json.loads(initial_overlay_manifest_path.read_text(encoding="utf-8"))
    initial_ternary_metrics = json.loads(initial_metrics_path.read_text(encoding="utf-8"))
    if (
        initial_overlay_manifest.get("base_revision") != manifest.revision
        or initial_overlay_manifest.get("base_model_id") != manifest.repo_id
    ):
        raise ValueError("initial ternary overlay targets a different pinned student base")
    if sha256(initial_ternary_predictions_path) != expected_initial_hash:
        raise ValueError("initial ternary validation prediction hash changed")
    if len(initial_predictions) != len(validation_examples):
        raise ValueError("initial ternary predictions do not cover the validation snapshot")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    (output_dir / "effective-config.yaml").write_bytes(config_path.read_bytes())
    run_metadata: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "state": "preflight_complete",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "config_sha256": sha256(config_path),
        "trainer_script_sha256": sha256(Path(__file__).resolve()),
        "teacher_cache_generator_script_sha256": sha256(repo / "scripts/cache_teacher_options.py"),
        "student_code_sha256": {
            name: sha256(repo / "src/tiny_omni_decision" / name)
            for name in (
                "student.py",
                "student_training.py",
                "student_eval.py",
                "student_evaluate.py",
                "ternary.py",
            )
        },
        "effective_config_path": str(output_dir / "effective-config.yaml"),
        "student_model_id": manifest.repo_id,
        "student_revision": manifest.revision,
        "student_model_manifest_sha256": sha256(student_manifest_path),
        "train_corpus_path": str(train_path),
        "train_corpus_sha256": sha256(train_path),
        "train_examples": len(train_examples),
        "train_id_order_sha256": order_hash,
        "sampler_policy": (
            "seeded weighted modality/source round-robin; max repeats 1; one full corpus pass"
        ),
        "sampling_counts_by_modality_source": planned_counts,
        "teacher_cache_path": str(cache_path),
        "teacher_cache_sha256": sha256(cache_path),
        "teacher_cache_generator_sha256": cache_metadata.get("cache_generator_script_sha256"),
        "teacher_cache_metadata_sha256": sha256(cache_metadata_path),
        "teacher_id": teacher_cfg["teacher_id"],
        "teacher_revision": teacher_cfg["teacher_revision"],
        "teacher_temperature": teacher_cfg["teacher_temperature"],
        "initial_ternary_overlay_manifest_sha256": sha256(initial_overlay_manifest_path),
        "initial_ternary_overlay_target_element_count": initial_overlay_manifest[
            "target_element_count"
        ],
        "validation_snapshot_sha256": sha256(validation_snapshot),
        "validation_manifest_sha256": sha256(validation_manifest),
        "validation_ids": len(validation_examples),
        "initial_ternary_predictions_sha256": expected_initial_hash,
        "initial_ternary_metrics_sha256": sha256(initial_metrics_path),
        "qat_quantization_device": "cuda",
        "cpu_overlay_vs_gpu_qat_comparison": (
            "paired validation comparison; FP32 group reductions can differ across CPU and CUDA"
        ),
        "loss": loss_cfg,
        "seed": train_cfg["seed"],
        "planned_steps": planned_steps,
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "optimizer": train_cfg["optimizer"],
        "optimizer_parameter_device": parameter_device_policy,
        "optimizer_state_dtype": student_cfg["ternary"]["optimizer_state_dtype"],
        "lr_scheduler": train_cfg["lr_scheduler"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "warmup_steps": max(1, round(planned_steps * float(train_cfg["warmup_ratio"]))),
        "checkpoint_selector": train_cfg["best_checkpoint_selector"],
        "started_at_utc": datetime.now(UTC).isoformat(),
        "device": torch.cuda.get_device_name(),
        "gpu_total_bytes": total_vram,
        "gpu_free_bytes_at_start": free_vram,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "transformers_version": importlib.metadata.version("transformers"),
        "package_versions": {
            distribution: importlib.metadata.version(distribution)
            for distribution in (
                "accelerate",
                "huggingface-hub",
                "numpy",
                "pydantic",
                "PyYAML",
                "safetensors",
                "tokenizers",
                "torchvision",
            )
        },
        "python_version": platform.python_version(),
        "cloud_compute_usd": 0,
        "sealed_audit_loaded": False,
        "teacher_artifacts_modified": False,
        "optimizer_resume_state_saved": False,
    }
    atomic_json(output_dir / "run-metadata.json", run_metadata)
    atomic_json(
        output_dir / "sampling-plan.json",
        {
            "seed": train_cfg["seed"],
            "sample_id_order_sha256": order_hash,
            "sample_ids": ordered_ids,
            "counts_by_modality_source": planned_counts,
            "unique_examples": len(set(ordered_ids)),
            "repeated_examples": len(ordered_ids) - len(set(ordered_ids)),
        },
    )

    random.seed(int(train_cfg["seed"]))
    torch.manual_seed(int(train_cfg["seed"]))
    torch.cuda.manual_seed_all(int(train_cfg["seed"]))
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda")
    target_names = apply_ternary_qat(
        model,
        group_size=int(student_cfg["ternary"]["group_size"]),
        threshold_factor=float(student_cfg["ternary"]["threshold_factor"]),
    )
    modules = dict(model.named_modules())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name in target_names:
        parent_name, _, leaf = name.rpartition(".")
        original = getattr(modules[parent_name].parametrizations, leaf).original
        original.requires_grad_(True)
        trainable.append(original)
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count != int(initial_overlay_manifest["target_element_count"]):
        raise ValueError(
            "QAT trainable parameter count differs from the verified ternary inventory"
        )
    if not trainable or not all(parameter.requires_grad for parameter in trainable):
        raise ValueError("verified ternary shadow parameters are not trainable")
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "parametrizations" not in name
    ):
        raise ValueError("a non-QAT parameter was unexpectedly left trainable")
    master_parameters = (
        make_fp32_cpu_master_parameters(trainable)
        if parameter_device_policy == "cpu_fp32_master"
        else trainable
    )
    run_metadata.update(
        {
            "state": "running",
            "target_parameter_tensors": len(target_names),
            "target_element_count": trainable_count,
            "target_parameter_names": list(target_names),
            "frozen_non_target_parameter_count": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if "parametrizations" not in name
            ),
            "max_sequence_length": train_cfg["max_sequence_length"],
            "trainable_dtype": "bfloat16 shadow weights; FP32 loss accumulation",
            "optimizer_master_dtype": (
                "float32" if parameter_device_policy == "cpu_fp32_master" else None
            ),
            "optimizer_master_device": (
                "cpu" if parameter_device_policy == "cpu_fp32_master" else "model"
            ),
            "optimizer_master_parameter_bytes": (
                sum(parameter.numel() * parameter.element_size() for parameter in master_parameters)
                if parameter_device_policy == "cpu_fp32_master"
                else 0
            ),
        }
    )
    atomic_json(output_dir / "run-metadata.json", run_metadata)

    optimizer = build_ternary_qat_optimizer(
        master_parameters,
        name=str(train_cfg["optimizer"]).lower(),
        learning_rate=float(train_cfg["learning_rate"]),
    )
    total_steps = planned_steps
    warmup_steps = max(1, round(total_steps * float(train_cfg["warmup_ratio"])))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    best_key: tuple[float, float, float] | None = None
    best_step = -1
    best_shadow_path = output_dir / "best-shadow.safetensors"
    history_path = output_dir / "training-history.jsonl"
    progress_path = output_dir / "latest-progress.json"
    evaluation_seconds = 0.0
    train_started = time.perf_counter()
    last_progress_time = train_started
    emit_progress(
        progress_path,
        event="training_start",
        step=0,
        examples_seen=0,
        total_steps=total_steps,
        elapsed_seconds=0.0,
    )
    consumed = 0
    rolling_losses: list[float] = []
    initial_pairing_verified: list[str] = []
    initial_probability_differences: list[float] = []
    initial_top1_matches: list[bool] = []

    def shadow_state() -> dict[str, torch.Tensor]:
        nonlocal modules
        state: dict[str, torch.Tensor] = {}
        for name in target_names:
            parent_name, _, leaf = name.rpartition(".")
            original = getattr(modules[parent_name].parametrizations, leaf).original
            state[name] = original.detach().cpu().contiguous()
        return state

    def save_shadow(path: Path) -> None:
        nonlocal shadow_state
        temporary = path.with_name(path.stem + ".tmp.safetensors")
        save_file(shadow_state(), str(temporary), metadata={"format": "bf16-qat-shadow-v1"})
        os.replace(temporary, path)

    def verify_initial_pairing(prediction: dict[str, Any]) -> None:
        index = len(initial_pairing_verified)
        reference = initial_predictions[index]
        if (
            prediction["sample_id"] != reference["sample_id"]
            or prediction["target"] != reference["target"]
            or prediction["options"] != reference["options"]
            or prediction["option_order_sha256"] != reference["option_order_sha256"]
        ):
            raise ValueError("step-0 predictions do not preserve the fixed validation pairing")
        differences = [
            abs(float(actual) - float(expected))
            for actual, expected in zip(
                prediction["option_probabilities"],
                reference["option_probabilities"],
                strict=True,
            )
        ]
        initial_probability_differences.extend(differences)
        initial_top1_matches.append(prediction["prediction"] == reference["prediction"])
        initial_pairing_verified.append(prediction["sample_id"])

    def evaluate_at(
        step: int,
        examples_seen: int,
        mean_loss: float | None,
        learning_rate_used: float | None,
        gradient_norm: float | None,
    ) -> None:
        nonlocal best_key, best_step, evaluation_seconds, run_metadata
        nonlocal modules, shadow_state, model, save_shadow, optimizer
        eval_started = time.perf_counter()
        model.eval()
        emit_progress(
            progress_path,
            event="validation_start",
            step=step,
            examples_seen=examples_seen,
            validation_examples=len(validation_examples),
            elapsed_seconds=eval_started - train_started,
        )
        # Each BF16 QAT shadow is held on CPU while its ternary value is used
        # in-place. This avoids 256 * 2 repeated full-model quantizations
        # without allocating a second full quantized model on the 16-GiB GPU.
        with cached_ternary_validation(model, target_names):
            metrics, predictions = evaluate_student_examples(
                model,
                processor,
                validation_examples,
                data_root=data_root,
                temperature=float(student_cfg["score_temperature"]),
                ece_bins=15,
                on_prediction=verify_initial_pairing if step == 0 else None,
            )
        emit_progress(
            progress_path,
            event="validation_done",
            step=step,
            examples_seen=examples_seen,
            validation_examples=len(validation_examples),
            validation_seconds=time.perf_counter() - eval_started,
            elapsed_seconds=time.perf_counter() - train_started,
        )
        teacher_comparison = compare_student_predictions(predictions, validation_teacher)
        if step == 0:
            if initial_pairing_verified != [prediction["sample_id"] for prediction in predictions]:
                raise ValueError("step-0 baseline comparison did not cover the full validation set")
            initial_comparison = {
                "reference": "existing CPU-packed ternary overlay validation",
                "candidate": "initial CUDA QAT forward from pinned BF16 base weights",
                "validation_ids": len(predictions),
                "sample_pairing_exact": True,
                "top1_prediction_agreement": sum(initial_top1_matches) / len(initial_top1_matches),
                "mean_absolute_option_probability_difference": (
                    sum(initial_probability_differences) / len(initial_probability_differences)
                ),
                "maximum_absolute_option_probability_difference": max(
                    initial_probability_differences
                ),
                "cpu_packed_validation_metrics": initial_ternary_metrics,
                "cuda_qat_validation_metrics": metrics,
                "quantization_reduction_note": (
                    "CUDA and CPU FP32 group reductions differ at BF16 threshold boundaries; "
                    "the existing CPU overlay is retained as a reference, while this run "
                    "uses CUDA QAT and validates its own packed export/reload path."
                ),
            }
            atomic_json(output_dir / "step-0-cpu-overlay-comparison.json", initial_comparison)
        key = (
            float(metrics["macro_modality"]["nll"]),
            -float(metrics["minimum_modality_accuracy"]),
            float(metrics["macro_modality"]["brier"]),
        )
        improved = best_key is None or key < best_key
        if improved:
            emit_progress(
                progress_path,
                event="best_shadow_save_start",
                step=step,
                examples_seen=examples_seen,
                elapsed_seconds=time.perf_counter() - train_started,
            )
            save_started = time.perf_counter()
            save_shadow(best_shadow_path)
            emit_progress(
                progress_path,
                event="best_shadow_save_done",
                step=step,
                examples_seen=examples_seen,
                save_seconds=time.perf_counter() - save_started,
                elapsed_seconds=time.perf_counter() - train_started,
            )
            best_key = key
            best_step = step
        eval_seconds = time.perf_counter() - eval_started
        evaluation_seconds += eval_seconds
        write_jsonl(output_dir / f"validation-step-{step:04d}.jsonl", predictions)
        atomic_json(
            output_dir / f"validation-step-{step:04d}.json",
            {
                "step": step,
                "examples_seen": examples_seen,
                "rolling_train_loss_mean": mean_loss,
                "learning_rate_used": learning_rate_used,
                "gradient_norm": gradient_norm,
                "validation_metrics": metrics,
                "teacher_comparison": teacher_comparison,
                "selector_key": key,
                "best_step": best_step,
                "is_best": improved,
                "evaluation_seconds": eval_seconds,
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
            },
        )
        record = {
            "step": step,
            "examples_seen": examples_seen,
            "unique_examples_seen": examples_seen,
            "rolling_train_loss_mean": mean_loss,
            "learning_rate_used": learning_rate_used,
            "gradient_norm": gradient_norm,
            "learning_rate_next": optimizer.param_groups[0]["lr"],
            "validation_macro_accuracy": metrics["macro_modality"]["accuracy"],
            "validation_minimum_modality_accuracy": metrics["minimum_modality_accuracy"],
            "validation_macro_nll": metrics["macro_modality"]["nll"],
            "validation_macro_brier": metrics["macro_modality"]["brier"],
            "validation_macro_ece": metrics["macro_modality"]["ece"],
            "validation_by_modality": {
                key_name: value
                for key_name, value in metrics.items()
                if key_name.startswith("modality:")
            },
            "selector_key": key,
            "best_step": best_step,
            "is_best": improved,
            "evaluation_seconds": eval_seconds,
            "elapsed_training_seconds": time.perf_counter() - train_started,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        run_metadata.update(
            {
                "global_step": step,
                "examples_consumed": examples_seen,
                "unique_examples_consumed": examples_seen,
                "best_validation_step": best_step,
                "best_selector_key": best_key,
                "best_shadow_sha256": sha256(best_shadow_path),
                "validation_evaluations": step // int(train_cfg["evaluation_interval"]) + 1,
                "last_validation_metrics": metrics,
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                "evaluation_seconds_total": evaluation_seconds,
            }
        )
        atomic_json(output_dir / "run-metadata.json", run_metadata)
        model.train()

    # QAT step 0 must match the separately exported ternary artifact before updates begin.
    evaluate_at(0, 0, None, None, None)
    accumulation = int(train_cfg["gradient_accumulation_steps"])
    interval = int(train_cfg["evaluation_interval"])
    max_grad_norm = float(train_cfg["max_gradient_norm"])
    for update_index, start in enumerate(range(0, len(order), accumulation), start=1):
        microbatch = order[start : start + accumulation]
        optimizer.zero_grad(set_to_none=True)
        if parameter_device_policy == "cpu_fp32_master":
            for parameter in trainable:
                parameter.grad = None
        interval_losses: list[float] = []
        for example in microbatch:
            query_inputs = processor_inputs_for_decision_example(
                processor, example, data_root=data_root
            )
            option_inputs = processor_inputs_for_options(processor, example.options)
            if max(query_inputs["input_ids"].shape[-1], option_inputs["input_ids"].shape[-1]) > int(
                train_cfg["max_sequence_length"]
            ):
                raise ValueError(f"{example.id}: sequence exceeds configured max length")
            query_embedding = model_sentence_embeddings(model, query_inputs)[0]
            option_embeddings = model_sentence_embeddings(model, option_inputs)
            logits = supplied_option_logits(
                query_embedding,
                option_embeddings,
                temperature=float(student_cfg["score_temperature"]),
            )
            cached = teacher_cache[example.id]
            teacher_probabilities = torch.tensor(
                cached["teacher_option_probabilities"],
                device=logits.device,
                dtype=torch.float32,
            )
            loss, loss_parts = student_option_distillation_loss(
                logits,
                teacher_probabilities,
                target_index=example.options.index(example.target),
                option_kl_weight=float(loss_cfg["option_kl"]),
                cross_entropy_weight=float(loss_cfg["cross_entropy"]),
                brier_weight=float(loss_cfg["brier"]),
            )
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite QAT loss at {example.id}")
            (loss / len(microbatch)).backward()
            interval_losses.append(float(loss.detach().cpu()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            max_grad_norm,
            error_if_nonfinite=True,
            foreach=False,
        )
        if not torch.isfinite(gradient_norm):
            raise ValueError(f"non-finite gradient norm at update {update_index}")
        current_lr = float(optimizer.param_groups[0]["lr"])
        if parameter_device_policy == "cpu_fp32_master":
            active_master_indices = copy_shadow_gradients_to_masters(
                trainable,
                master_parameters,
                check_finite=False,
            )
            if not active_master_indices:
                raise RuntimeError(f"no QAT shadow gradients at update {update_index}")
        else:
            active_master_indices = []
        optimizer.step()
        if parameter_device_policy == "cpu_fp32_master":
            sync_cpu_masters_to_shadows(
                trainable,
                master_parameters,
                active_master_indices,
            )
            for parameter in trainable:
                parameter.grad = None
        scheduler.step()
        consumed += len(microbatch)
        rolling_losses.extend(interval_losses)
        if update_index == 1 or update_index % 16 == 0 or update_index == total_steps:
            now = time.perf_counter()
            emit_progress(
                progress_path,
                event="training_update",
                step=update_index,
                examples_seen=consumed,
                total_steps=total_steps,
                window_seconds=now - last_progress_time,
                elapsed_seconds=now - train_started,
                microbatch_loss_mean=sum(interval_losses) / len(interval_losses),
                allocated_vram_bytes=torch.cuda.memory_allocated(),
                reserved_vram_bytes=torch.cuda.memory_reserved(),
            )
            last_progress_time = now
        if update_index % interval == 0 or update_index == total_steps:
            mean_loss = sum(rolling_losses) / len(rolling_losses) if rolling_losses else None
            evaluate_at(update_index, consumed, mean_loss, current_lr, float(gradient_norm))
            rolling_losses.clear()
        if update_index % int(train_cfg["checkpoint_interval"]) == 0 or update_index == total_steps:
            with history_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": "optimizer_update",
                            "step": update_index,
                            "examples_consumed": consumed,
                            "learning_rate_used": current_lr,
                            "loss_mean_microbatch": sum(interval_losses) / len(interval_losses),
                            "gradient_norm": float(gradient_norm.detach().cpu()),
                            "elapsed_training_seconds": time.perf_counter() - train_started,
                            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
    if consumed != len(train_examples):
        raise RuntimeError(f"training consumed {consumed} examples, expected {len(train_examples)}")

    training_loop_seconds = time.perf_counter() - train_started
    final_shadow_path = output_dir / "final-shadow.safetensors"
    save_shadow(final_shadow_path)
    final_shadow_sha256 = sha256(final_shadow_path)
    del optimizer, scheduler, model, modules, trainable, shadow_state, save_shadow, evaluate_at
    torch.cuda.empty_cache()

    export_reload_started = time.perf_counter()
    overlay_root = output_dir / "overlays"
    overlay_root.mkdir()
    model_id = manifest.repo_id
    revision = manifest.revision
    shadow_paths = (("best", best_shadow_path), ("final", final_shadow_path))
    reload_results: dict[str, Any] = {}
    for label, shadow_path in shadow_paths:
        weights = load_file(str(shadow_path), device="cpu")
        export_model = AutoModel.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to("cuda")
        params = dict(export_model.named_parameters())
        with torch.no_grad():
            for name in target_names:
                parameter = params[name]
                parameter.copy_(weights[name].to(device=parameter.device, dtype=parameter.dtype))
        overlay_dir = overlay_root / label
        export_packed_ternary_overlay(
            export_model,
            overlay_dir,
            base_model_id=model_id,
            base_revision=revision,
            group_size=int(student_cfg["ternary"]["group_size"]),
            threshold_factor=float(student_cfg["ternary"]["threshold_factor"]),
        )
        del export_model, params, weights
        torch.cuda.empty_cache()
        reload_model = (
            AutoModel.from_pretrained(
                str(model_path),
                dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            .to("cuda")
            .eval()
        )
        loaded_overlay = load_packed_ternary_overlay(
            reload_model,
            overlay_dir,
            expected_base_model_id=model_id,
            expected_base_revision=revision,
        )
        reload_metrics, reload_predictions = evaluate_student_examples(
            reload_model,
            processor,
            validation_examples,
            data_root=data_root,
            temperature=float(student_cfg["score_temperature"]),
            ece_bins=15,
        )
        expected_step = best_step if label == "best" else total_steps
        expected_path = output_dir / f"validation-step-{expected_step:04d}.jsonl"
        expected_predictions = [
            json.loads(line)
            for line in expected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(expected_predictions) != len(reload_predictions):
            raise ValueError(f"{label} overlay reload changed the validation prediction count")
        reload_probability_differences: list[float] = []
        for expected, actual in zip(expected_predictions, reload_predictions, strict=True):
            if any(
                expected[field] != actual[field]
                for field in ("sample_id", "target", "options", "option_order_sha256")
            ):
                raise ValueError(f"{label} overlay reload changed validation pairing")
            reload_probability_differences.extend(
                abs(float(left) - float(right))
                for left, right in zip(
                    expected["option_probabilities"],
                    actual["option_probabilities"],
                    strict=True,
                )
            )
        max_reload_probability_difference = max(reload_probability_differences)
        if max_reload_probability_difference > 1e-5:
            raise ValueError(
                f"{label} overlay reload differs from step {expected_step} QAT forward: "
                f"max probability abs difference {max_reload_probability_difference:.8g}"
            )
        prediction_path = output_dir / f"{label}-reload-predictions.jsonl"
        metrics_path = output_dir / f"{label}-reload-metrics.json"
        write_jsonl(prediction_path, reload_predictions)
        atomic_json(
            metrics_path,
            {
                "validation_metrics": reload_metrics,
                "overlay_manifest_sha256": sha256(overlay_dir / "manifest.json"),
                "overlay_tensor_sha256": loaded_overlay["tensor_file_sha256"],
                "overlay_tensor_bytes": loaded_overlay["tensor_file_bytes"],
                "predictions_sha256": sha256(prediction_path),
                "paired_qat_step": expected_step,
                "max_qat_reload_probability_abs_difference": max_reload_probability_difference,
            },
        )
        reload_results[label] = {
            "overlay_manifest_sha256": sha256(overlay_dir / "manifest.json"),
            "overlay_tensor_sha256": loaded_overlay["tensor_file_sha256"],
            "overlay_tensor_bytes": loaded_overlay["tensor_file_bytes"],
            "predictions_sha256": sha256(prediction_path),
            "metrics_sha256": sha256(metrics_path),
            "paired_qat_step": expected_step,
            "max_qat_reload_probability_abs_difference": max_reload_probability_difference,
            "validation_metrics": reload_metrics,
        }
        del reload_model
        torch.cuda.empty_cache()

    run_metadata.update(
        {
            "state": "complete",
            "ended_at_utc": datetime.now(UTC).isoformat(),
            "global_step": total_steps,
            "examples_consumed": consumed,
            "unique_examples_consumed": len(set(ordered_ids)),
            "best_validation_step": best_step,
            "best_shadow_sha256": sha256(best_shadow_path),
            "final_shadow_sha256": final_shadow_sha256,
            "best_checkpoint_selector": train_cfg["best_checkpoint_selector"],
            "training_loop_seconds": training_loop_seconds,
            "export_reload_wall_seconds": time.perf_counter() - export_reload_started,
            "total_wall_seconds": time.perf_counter() - train_started,
            "evaluation_seconds_total": evaluation_seconds,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
            "packed_overlay_reload_results": reload_results,
            "sealed_audit_loaded": False,
            "product_teacher_promotion": False,
        }
    )
    atomic_json(output_dir / "run-metadata.json", run_metadata)
    print(f"ternary QAT run complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
