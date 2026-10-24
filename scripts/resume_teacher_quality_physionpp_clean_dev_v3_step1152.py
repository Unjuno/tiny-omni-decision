from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import run_teacher_quality_physionpp_clean_dev_v3 as original

from tiny_omni_decision.trainer import (
    experiment_id_from_resume_checkpoint,
    file_sha256,
    resolve_resume_checkpoint,
    sample_id_order_sha256,
)
from tiny_omni_decision.training import decision_training_config, deterministic_sample_order

GENERATION = original.CORPUS.parent
FAILED_OUTPUT = GENERATION / "run-seed17-2048-resume-step256-exact"
RESUME_FROM = (
    FAILED_OUTPUT
    / "checkpoints"
    / "resume-step-001152-6e58a0c910894db1b9c89434de59556e"
)
OUTPUT = GENERATION / "run-seed17-2048-resume-step1152-exact"


def _snapshot_hashes(checkpoint: Path) -> dict[str, dict[str, Any]]:
    return {
        item.relative_to(checkpoint).as_posix(): {
            "size_bytes": item.stat().st_size,
            "sha256": file_sha256(str(item)),
        }
        for item in sorted(path for path in checkpoint.rglob("*") if path.is_file())
    }


def _preflight() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite continuation output: {OUTPUT}")
    checkpoint = resolve_resume_checkpoint(RESUME_FROM)
    state = json.loads((checkpoint / "trainer-state.json").read_text(encoding="utf-8"))
    original_invocation = json.loads(
        (FAILED_OUTPUT / "run-invocation.json").read_text(encoding="utf-8")
    )
    refs = original_invocation["reference_adapters"]
    reference_adapter = Path(refs["teacher_v1"]).resolve()
    e_long_adapter = Path(refs["e_long_selected"]).resolve()
    if refs.get("usage") != "read-only evaluation only":
        raise ValueError("reference adapters were not recorded as evaluation-only")

    preflight = original._preflight(reference_adapter, e_long_adapter)
    config = decision_training_config(preflight["config"])
    expected_hashes = {
        "train_rows_sha256": file_sha256(str(original.CORPUS / "train.jsonl")),
        "validation_rows_sha256": file_sha256(str(original.CORPUS / "validation.jsonl")),
        "config_sha256": preflight["config_sha256"],
        "model_manifest_sha256": file_sha256(str(original.BASE_MANIFEST)),
        "trainer_code_sha256": file_sha256(
            str(Path(original.run_training.__globals__["__file__"]))
        ),
        "training_code_sha256": file_sha256(
            str(Path(original.run_training.__globals__["__file__"]).with_name("training.py"))
        ),
    }
    for key, expected in expected_hashes.items():
        if state.get(key) != expected:
            raise ValueError(f"step-1152 resume snapshot {key} differs from frozen inputs")

    effective_payload = {
        "raw_config": preflight["config"],
        "decision_training_config": config.model_dump(mode="json"),
        "modalities": ["audio", "image", "text", "video"],
        "reference_adapter_path": str(reference_adapter),
        "model_manifest_path": str(original.BASE_MANIFEST),
        "tiny_overfit": False,
    }
    effective_hash = hashlib.sha256(
        json.dumps(effective_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if state.get("effective_config_sha256") != effective_hash:
        raise ValueError("effective config hash differs from the exact resume state")
    if state.get("global_step") != 1152 or state.get("sample_index") != 4608:
        raise ValueError("resume point is not the exact step-1152 / sample-4,608 snapshot")
    if state.get("best_step") != 1152:
        raise ValueError("selector state in the resume snapshot is not the recorded best step")
    if state.get("rng_state") is None:
        raise ValueError("resume snapshot does not contain all required RNG state")

    train = original._read(original.CORPUS / "train.jsonl")
    sample_order, _ = deterministic_sample_order(
        train,
        seed=config.seed,
        limit=config.max_train_examples,
        modality_weights=config.modality_weights,
        source_weights=config.source_weights,
        max_sample_repeats=config.max_sample_repeats,
        video_task_weights=config.video_task_weights or None,
    )
    consumed = sample_order[: state["sample_index"]]
    if sample_id_order_sha256(consumed) != state.get("consumed_sample_id_order_sha256"):
        raise ValueError("replayed consumed sample IDs differ from the saved resume hash")
    if dict(Counter(f"{row.modality}:{row.source}" for row in consumed)) != state.get(
        "consumed", {}
    ):
        raise ValueError("replayed consumed source/modality counts differ from saved state")
    task_counts = Counter(row.task_type for row in consumed if row.task_type is not None)
    if dict(task_counts) != state.get("consumed_task_types", {}):
        raise ValueError("replayed video task counts differ from saved state")

    required_files = (
        "optimizer.pt",
        "scheduler.pt",
        "best/adapter_model.safetensors",
        "best-validation-predictions.jsonl",
        "best-checkpoint.json",
    )
    for name in required_files:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"exact resume snapshot is missing {name}")
    best_meta = json.loads((checkpoint / "best-checkpoint.json").read_text(encoding="utf-8"))
    adapter_hash = file_sha256(str(checkpoint / "best/adapter_model.safetensors"))
    if best_meta.get("adapter_sha256") != adapter_hash:
        raise ValueError("selected adapter hash differs from snapshot metadata")
    if best_meta.get("step") != 1152:
        raise ValueError("selected adapter in exact snapshot is not step 1152")

    failed_ledger = [
        json.loads(line)
        for line in (FAILED_OUTPUT / "experiments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    training_id = experiment_id_from_resume_checkpoint(checkpoint)[1]
    events = [event for event in failed_ledger if event.get("experiment_id") == training_id]
    if not any(event.get("status") == "started" for event in events):
        raise ValueError("exact resume checkpoint does not belong to a logged run")
    if any(event.get("status") == "completed" for event in events):
        raise ValueError("run ledger already records this experiment as completed")
    if not any(event.get("status") == "failed" for event in events):
        raise ValueError("failed parent attempt is not recorded in the run ledger")

    selector = original.deterministic_validation_subset(
        original._read(original.CORPUS / "validation.jsonl"),
        seed=config.seed,
        limit=config.selection_eval_examples,
        video_task_weights=config.video_task_weights or None,
    )
    selector_hash = original._hash_ids(selector)
    expected_selector_hash = original_invocation["validation_selector_id_order_sha256"]
    if selector_hash != expected_selector_hash:
        raise ValueError("validation selector IDs/order differ from the failed attempt")
    preflight.update(
        {
            "artifact_output": str(OUTPUT),
            "initialization": (
                "exact continuation from verified step-1152 snapshot after "
                "checkpoint serialization failure at step 1280"
            ),
            "new_seed_or_hyperparameter_run": False,
            "resume_from": str(checkpoint),
            "resume_global_step": state["global_step"],
            "resume_sample_index": state["sample_index"],
            "resume_training_experiment_id": training_id,
            "resume_consumed_id_order_sha256": state["consumed_sample_id_order_sha256"],
            "effective_config_sha256": effective_hash,
            "resume_snapshot_files": _snapshot_hashes(checkpoint),
            "exact_resume_checks": {
                "config_corpus_manifest_and_code_hashes": "PASS",
                "effective_config_hash": "PASS",
                "optimizer_scheduler_adapter_and_rng_presence": "PASS",
                "sampler_order_and_consumed_counts": "PASS",
                "selector_best_adapter_hash": "PASS",
                "validation_selector_order": "PASS",
            },
            "failed_parent_output": str(FAILED_OUTPUT),
            "failed_step_1280_metrics_preserved": True,
            "sealed_audit_loaded": False,
        }
    )
    return preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact-resume clean-dev-v3 from step 1152.")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    preflight = _preflight()
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    OUTPUT.mkdir(parents=True, exist_ok=False)
    selector = original.deterministic_validation_subset(
        original._read(original.CORPUS / "validation.jsonl"),
        seed=17,
        limit=2048,
        video_task_weights=decision_training_config(preflight["config"]).video_task_weights,
    )
    for name, payload in (
        ("effective-config.json", preflight["config"]),
        ("run-invocation.json", preflight),
        (
            "resume-preflight.json",
            {
                "checks": preflight["exact_resume_checks"],
                "resume_from": preflight["resume_from"],
                "global_step": preflight["resume_global_step"],
                "sample_index": preflight["resume_sample_index"],
                "consumed_id_order_sha256": preflight["resume_consumed_id_order_sha256"],
                "checkpoint_files": preflight["resume_snapshot_files"],
            },
        ),
        (
            "validation-selector.json",
            {
                "corpus_sha256": preflight["validation_sha256"],
                "id_order_sha256": preflight["validation_selector_id_order_sha256"],
                "ids": [row.id for row in selector],
            },
        ),
    ):
        (OUTPUT / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    started = time.monotonic()
    try:
        result = original.run_training(
            train_path=original.CORPUS / "train.jsonl",
            validation_path=original.CORPUS / "validation.jsonl",
            config_path=original.CONFIG,
            output_dir=OUTPUT,
            reference_adapter_path=Path(
                preflight["reference_adapters"]["teacher_v1"]
            ).resolve(),
            model_manifest_path=original.BASE_MANIFEST,
            media_root=original.HARDLINK_MEDIA_ROOT,
            resume_from=Path(preflight["resume_from"]),
        )
    except Exception as error:
        status = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_seconds": time.monotonic() - started,
            "last_exact_resume_step": preflight["resume_global_step"],
            "experiment_id": preflight["experiment_id"],
        }
        (OUTPUT / "continuation-status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    summary = {
        "experiment_id": preflight["experiment_id"],
        "training_experiment_id": preflight["resume_training_experiment_id"],
        "status": "completed_training" if result["global_steps"] == 2048 else "incomplete",
        "global_steps": result["global_steps"],
        "resume_global_step": preflight["resume_global_step"],
        "best_checkpoint_step": result["best_checkpoint_step"],
        "validation_best_metrics": result["validation_best_metrics"],
        "checkpoint_reload_verified": result["checkpoint_reload_verified"],
        "training_seconds": result["training_seconds"],
        "wall_seconds": time.monotonic() - started,
        "peak_vram_bytes": result["max_allocated_vram_bytes"],
        "selected_adapter_sha256": result.get("best_adapter_sha256"),
        "train_corpus_sha256": preflight["train_sha256"],
        "validation_corpus_sha256": preflight["validation_sha256"],
    }
    (OUTPUT / "run-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
