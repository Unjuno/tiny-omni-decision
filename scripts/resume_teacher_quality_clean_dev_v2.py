from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.io import load_structured_file  # noqa: E402
from tiny_omni_decision.schema import BaseModelManifest  # noqa: E402
from tiny_omni_decision.trainer import (  # noqa: E402
    _eligible,
    _read_examples,
    resolve_resume_checkpoint,
    run_training,
    sample_id_order_sha256,
)
from tiny_omni_decision.training import (  # noqa: E402
    decision_training_config,
    deterministic_sample_order,
    deterministic_validation_subset,
)

RUN = ROOT / "artifacts/tiny-omni-decision-teacher-v2/clean-dev-v2-seed17-2048"
RESUME = RUN / "checkpoints/resume-step-000896-ce5b44dfa4914c019c28ac8a11c40b2a"
OUTPUT = ROOT / "artifacts/tiny-omni-decision-teacher-v2/clean-dev-v2-seed17-2048-resume-step896"
TRAIN = ROOT / "data/processed/teacher-quality-next/clean-dev-v2/train.jsonl"
VALIDATION = ROOT / "data/processed/teacher-quality-next/clean-dev-v2/validation.jsonl"
CONFIG = ROOT / "configs/decision/teacher_quality_clean_dev_v2.yaml"
BASE_MANIFEST = ROOT / "manifests/base-model.example.yaml"
REFERENCE_ADAPTER = ROOT / "artifacts/tiny-omni-decision-teacher-v1/selected"
PREVIOUS_OUTPUT: Path | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight() -> dict[str, object]:
    if (
        subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        != "codex/teacher-quality-clean-dev-v1"
    ):
        raise ValueError("resume must use the original clean-dev experiment branch")
    global RESUME
    RESUME = resolve_resume_checkpoint(RESUME)
    state_path = RESUME / "trainer-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not 0 < int(state.get("global_step", -1)) < 2048:
        raise ValueError("resume checkpoint must be inside the original 2,048-step run")
    required_rng_state = {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }
    if not isinstance(state.get("rng_state"), dict) or not required_rng_state <= set(
        state["rng_state"]
    ):
        raise ValueError("resume snapshot is missing one or more random-number-generator states")
    expected = {
        "train_rows_sha256": sha256_file(TRAIN),
        "validation_rows_sha256": sha256_file(VALIDATION),
        "config_sha256": sha256_file(CONFIG),
        "trainer_code_sha256": sha256_file(ROOT / "src/tiny_omni_decision/trainer.py"),
        "training_code_sha256": sha256_file(ROOT / "src/tiny_omni_decision/training.py"),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"resume snapshot mismatch for {key}")
    if not all((RESUME / name).is_file() for name in ("optimizer.pt", "scheduler.pt")):
        raise ValueError("resume snapshot is missing optimizer or scheduler state")
    if not state.get("rng_state") or not (RESUME / "best").is_dir():
        raise ValueError("resume snapshot is missing RNG state or its selected best adapter")
    invocation_path = RUN / "run-invocation.json"
    preflight_path = RUN / "sampler-preflight.json"
    selector_path = RUN / "validation-selector.json"
    for path in (invocation_path, preflight_path, selector_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    if invocation["config"]["training"]["max_steps"] != 2048:
        raise ValueError("the original run is not configured for the required 2,048 updates")
    if invocation["config"]["training"]["seed"] != 17:
        raise ValueError("the original run seed differs from the pinned seed 17")
    if invocation["config_sha256"] != state["config_sha256"]:
        raise ValueError("original invocation config hash differs from resume state")
    if invocation["validation_sha256"] != state["validation_rows_sha256"]:
        raise ValueError("original validation hash differs from resume state")
    if invocation["train_sha256"] != state["train_rows_sha256"]:
        raise ValueError("original training hash differs from resume state")

    raw_config = load_structured_file(CONFIG)
    config = decision_training_config(raw_config)
    manifest = BaseModelManifest.model_validate(load_structured_file(BASE_MANIFEST))
    effective_payload = {
        "raw_config": raw_config,
        "decision_training_config": config.model_dump(mode="json"),
        "modalities": ["audio", "image", "text", "video"],
        "reference_adapter_path": str(REFERENCE_ADAPTER),
        "model_manifest_path": str(BASE_MANIFEST),
        "tiny_overfit": False,
    }
    effective_hash = hashlib.sha256(
        json.dumps(effective_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if effective_hash != state.get("effective_config_sha256"):
        raise ValueError("effective config hash differs from the exact resume snapshot")
    if sha256_file(BASE_MANIFEST) != state.get("model_manifest_sha256"):
        raise ValueError("base-model manifest hash differs from the exact resume snapshot")
    if manifest.revision != state.get("base_revision") or manifest.processor_revision != state.get(
        "processor_revision"
    ):
        raise ValueError("base or processor revision differs from the exact resume snapshot")

    train_rows, _ = _eligible(_read_examples(TRAIN), {"text", "image", "audio", "video"})
    validation_rows, _ = _eligible(_read_examples(VALIDATION), {"text", "image", "audio", "video"})
    order, _ = deterministic_sample_order(
        train_rows,
        seed=config.seed,
        limit=config.max_train_examples,
        modality_weights=config.modality_weights,
        source_weights=config.source_weights,
        max_sample_repeats=config.max_sample_repeats,
        video_task_weights=config.video_task_weights or None,
    )
    consumed = order[: int(state["sample_index"])]
    if sample_id_order_sha256(consumed) != state.get("consumed_sample_id_order_sha256"):
        raise ValueError("sampler replay does not match consumed IDs in resume snapshot")
    selector = deterministic_validation_subset(
        validation_rows,
        seed=17,
        limit=min(config.selection_eval_examples, len(validation_rows)),
        video_task_weights=config.video_task_weights or None,
    )
    original_selector = json.loads(selector_path.read_text(encoding="utf-8"))
    if [row.id for row in selector] != original_selector.get("ids"):
        raise ValueError("validation selector IDs/order differ from the frozen run")
    current_best = json.loads((RUN / "best-checkpoint.json").read_text(encoding="utf-8"))
    resume_best = json.loads((RESUME / "best-checkpoint.json").read_text(encoding="utf-8"))
    if resume_best.get("step") != state["best_step"]:
        raise ValueError("resume snapshot selector/best-step state does not match its metadata")
    best_weights = RESUME / "best/adapter_model.safetensors"
    if not best_weights.is_file():
        best_weights = RESUME / "best/adapter_model.bin"
    if sha256_file(best_weights) != resume_best.get("adapter_sha256"):
        raise ValueError("resume snapshot selected adapter hash differs from its metadata")
    import torch

    optimizer_state = torch.load(RESUME / "optimizer.pt", map_location="cpu", weights_only=False)
    scheduler_state = torch.load(RESUME / "scheduler.pt", map_location="cpu", weights_only=False)
    optimizer_steps = [
        int(value.item() if hasattr(value, "item") else value)
        for entry in optimizer_state.get("state", {}).values()
        if (value := entry.get("step")) is not None
    ]
    if not optimizer_steps or set(optimizer_steps) != {int(state["global_step"])}:
        raise ValueError("optimizer state step counters do not match the resume global step")
    if int(scheduler_state.get("last_epoch", -1)) != int(state["global_step"]):
        raise ValueError("scheduler state does not match the resume global step")
    if int(scheduler_state.get("_step_count", -1)) != int(state["global_step"]) + 1:
        raise ValueError("scheduler update count does not match the resume global step")
    checkpoint_files = (
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer-state.json",
        "best/adapter_model.safetensors",
        "best-checkpoint.json",
        "best-validation-predictions.jsonl",
    )
    snapshot_hashes = {
        name: sha256_file(RESUME / name)
        for name in checkpoint_files
        if (RESUME / name).is_file()
    }
    return {
        "resume_step": state["global_step"],
        "resume_sample_index": state["sample_index"],
        "planned_max_steps": config.max_steps,
        "remaining_optimizer_steps": config.max_steps - int(state["global_step"]),
        "early_stopping_patience": config.early_stopping_patience,
        "evaluations_without_improvement": state["evaluations_without_improvement"],
        "effective_config_sha256": effective_hash,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "replayed_consumed_examples": len(consumed),
        "consumed_id_order_sha256": state["consumed_sample_id_order_sha256"],
        "validation_selector_id_order_sha256": sample_id_order_sha256(selector),
        "best_at_resume_step": resume_best["step"],
        "original_run_best_step_at_failure": current_best["step"],
        "optimizer_state_entries": len(optimizer_state.get("state", {})),
        "optimizer_steps": sorted(set(optimizer_steps)),
        "scheduler_last_epoch": scheduler_state["last_epoch"],
        "scheduler_step_count": scheduler_state["_step_count"],
        "scheduler_last_lr": scheduler_state.get("_last_lr"),
        "snapshot_sha256": snapshot_hashes,
        "resume_hashes_match": True,
    }


def run() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite continuation artifact: {OUTPUT}")
    checks = preflight()
    state = json.loads((RESUME / "trainer-state.json").read_text(encoding="utf-8"))
    invocation_path = RUN / "run-invocation.json"
    preflight_path = RUN / "sampler-preflight.json"
    selector_path = RUN / "validation-selector.json"

    OUTPUT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(invocation_path, OUTPUT / "parent-run-invocation.json")
    shutil.copy2(preflight_path, OUTPUT / "sampler-preflight.json")
    shutil.copy2(selector_path, OUTPUT / "validation-selector.json")
    previous_attempt = None
    if PREVIOUS_OUTPUT is not None:
        if not PREVIOUS_OUTPUT.is_dir():
            raise FileNotFoundError(PREVIOUS_OUTPUT)
        previous_history = PREVIOUS_OUTPUT / "training-history.jsonl"
        logged_steps: list[int] = []
        if previous_history.is_file():
            retained_lines = []
            for line in previous_history.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                logged_steps.append(int(record["step"]))
                if int(record["step"]) <= int(checks["resume_step"]):
                    retained_lines.append(json.dumps(record, sort_keys=True))
            (OUTPUT / "training-history.jsonl").write_text(
                "\n".join(retained_lines) + ("\n" if retained_lines else ""),
                encoding="utf-8",
            )
        copied = []
        for source in PREVIOUS_OUTPUT.iterdir():
            if not source.is_file():
                continue
            step_match = re.fullmatch(
                r"validation-step-(\d+)(?:-predictions)?\.jsonl?", source.name
            )
            is_step_evidence = step_match is not None and int(step_match.group(1)) <= int(
                checks["resume_step"]
            )
            is_small_run_evidence = source.name in {
                "validation-baseline-evaluation.json",
                "validation-baseline-predictions.jsonl",
                "fixed-step-comparison-512-1024.json",
                "learning-curve-comparison-512-1024-1152.json",
                "video-accuracy-paired-scene-bootstrap-512-1024.json",
            }
            if is_step_evidence or is_small_run_evidence:
                shutil.copy2(source, OUTPUT / source.name)
                copied.append(source.name)
        previous_attempt = {
            "output": str(PREVIOUS_OUTPUT),
            "last_logged_step": max(logged_steps) if logged_steps else None,
            "resume_snapshot_step": checks["resume_step"],
            "validation_at_last_logged_step_exists": (
                PREVIOUS_OUTPUT / f"validation-step-{max(logged_steps)}.json"
            ).is_file()
            if logged_steps
            else False,
            "interruption_reason": "UNKNOWN; process ended without a persisted terminal record",
            "copied_parent_evidence": sorted(copied),
        }
    failure_event = next(
        (
            json.loads(line)
            for line in (RUN / "experiments.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("status") == "failed"
        ),
        None,
    )
    continuation = {
        "continuation_id": OUTPUT.name,
        "parent_experiment_id": state["experiment_id"],
        "parent_artifact": str(RUN),
        "resume_snapshot": str(RESUME),
        "resume_global_step": state["global_step"],
        "resume_sample_index": state["sample_index"],
        "resume_consumed_id_order_sha256": state["consumed_sample_id_order_sha256"],
        "resume_best_step": state["best_step"],
        "resume_started_at_utc": datetime.now(UTC).isoformat(),
        "original_run_failure_preserved": failure_event,
        "original_artifacts_are_read_only": True,
        "new_seed_or_hyperparameter_run": False,
        "sealed_audit_loaded": False,
        "preflight": checks,
        "interrupted_previous_continuation": previous_attempt,
    }
    (OUTPUT / "continuation-manifest.json").write_text(
        json.dumps(continuation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = run_training(
        train_path=TRAIN,
        validation_path=VALIDATION,
        config_path=CONFIG,
        output_dir=OUTPUT,
        reference_adapter_path=REFERENCE_ADAPTER,
        model_manifest_path=BASE_MANIFEST,
        media_root=ROOT / "data",
        resume_from=RESUME,
    )
    summary = {
        "experiment_id": result["experiment_id"],
        "global_steps": result["global_steps"],
        "best_checkpoint_step": result["best_checkpoint_step"],
        "validation_best_metrics": result["validation_best_metrics"],
        "checkpoint_reload_verified": result["checkpoint_reload_verified"],
        "training_seconds": result["training_seconds"],
        "wall_seconds": result["wall_seconds"],
        "peak_vram_bytes": result["max_allocated_vram_bytes"],
    }
    (OUTPUT / "continuation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume the frozen clean-dev-v2 run exactly.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=RESUME)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--previous-output", type=Path)
    args = parser.parse_args()
    RESUME = args.resume_from
    OUTPUT = args.output
    PREVIOUS_OUTPUT = args.previous_output
    if args.preflight_only:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    else:
        print(json.dumps(run(), indent=2, sort_keys=True))
