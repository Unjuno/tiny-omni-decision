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
PARENT_OUTPUT = original.OUTPUT
RESUME_FROM = (
    PARENT_OUTPUT
    / "checkpoints"
    / "resume-step-000256-f2de92412b93410e8af2e6339f4b300b"
)
OUTPUT = GENERATION / "run-seed17-2048-resume-step256-exact"
ATTEMPT_STATUS = PARENT_OUTPUT / "attempt-status.json"


def _checkpoint_file_hashes(checkpoint: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        relative = path.relative_to(checkpoint).as_posix()
        result[relative] = {"size_bytes": path.stat().st_size, "sha256": file_sha256(str(path))}
    return result


def _preflight(
    reference_adapter: Path, e_long_adapter: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite resume output: {OUTPUT}")
    if not ATTEMPT_STATUS.is_file():
        raise FileNotFoundError(f"parent stop record is missing: {ATTEMPT_STATUS}")
    stop = json.loads(ATTEMPT_STATUS.read_text(encoding="utf-8"))
    if stop.get("status") != "blocked_resource_limit":
        raise ValueError("parent run is not recorded as a resource-limited stop")
    if stop.get("sealed_audit_loaded") is not False:
        raise ValueError("parent stop record does not affirm sealed-audit isolation")

    preflight = original._preflight(reference_adapter.resolve(), e_long_adapter.resolve())
    checkpoint = resolve_resume_checkpoint(RESUME_FROM)
    state = json.loads((checkpoint / "trainer-state.json").read_text(encoding="utf-8"))
    config = decision_training_config(preflight["config"])
    train_path = original.CORPUS / "train.jsonl"
    validation_path = original.CORPUS / "validation.jsonl"
    manifest_path = original.BASE_MANIFEST

    trainer_path = Path(original.run_training.__globals__["__file__"])
    expected_code_hashes = {
        "trainer_code_sha256": file_sha256(str(trainer_path)),
        "training_code_sha256": file_sha256(
            str(trainer_path.with_name("training.py"))
        ),
    }
    expected_hashes = {
        "train_rows_sha256": file_sha256(str(train_path)),
        "validation_rows_sha256": file_sha256(str(validation_path)),
        "config_sha256": preflight["config_sha256"],
        "model_manifest_sha256": file_sha256(str(manifest_path)),
        **expected_code_hashes,
    }
    for key, expected in expected_hashes.items():
        if state.get(key) != expected:
            raise ValueError(f"resume state {key} differs from current frozen input")

    effective_payload = {
        "raw_config": preflight["config"],
        "decision_training_config": config.model_dump(mode="json"),
        "modalities": ["audio", "image", "text", "video"],
        "reference_adapter_path": str(reference_adapter.resolve()),
        "model_manifest_path": str(manifest_path),
        "tiny_overfit": False,
    }
    effective_hash = hashlib.sha256(
        json.dumps(effective_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if state.get("effective_config_sha256") != effective_hash:
        raise ValueError("resume state effective config hash differs from the original run")

    if state.get("global_step") != 256 or state.get("sample_index") != 1024:
        raise ValueError("resume snapshot is not the exact step-256 / sample-1,024 state")
    if state.get("best_step") != 256:
        raise ValueError("resume snapshot selector state is not the saved step-256 best")
    training_experiment_id = experiment_id_from_resume_checkpoint(checkpoint)[1]
    ledger_path = PARENT_OUTPUT / "experiments.jsonl"
    ledger = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matching_events = [
        event for event in ledger if event.get("experiment_id") == training_experiment_id
    ]
    if not any(event.get("status") == "started" for event in matching_events):
        raise ValueError("resume checkpoint ID is absent from the parent run ledger")
    if any(event.get("status") == "completed" for event in matching_events):
        raise ValueError("parent run ledger already marks this experiment completed")
    if stop.get("experiment_id") != preflight.get("experiment_id"):
        raise ValueError("safety-stop record differs from the configured run identity")

    train = original._read(train_path)
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
        raise ValueError("reconstructed consumed sample order differs from resume state")
    consumed_counts = Counter(f"{row.modality}:{row.source}" for row in consumed)
    if dict(consumed_counts) != state.get("consumed", {}):
        raise ValueError("reconstructed source/modality counts differ from resume state")
    task_counts = Counter(row.task_type for row in consumed if row.task_type is not None)
    if dict(task_counts) != state.get("consumed_task_types", {}):
        raise ValueError("reconstructed question-type counts differ from resume state")

    for name in ("optimizer.pt", "scheduler.pt", "best/adapter_model.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"resume snapshot is missing {name}")
    best_meta = json.loads((checkpoint / "best-checkpoint.json").read_text(encoding="utf-8"))
    if best_meta.get("adapter_sha256") != file_sha256(
        str(checkpoint / "best" / "adapter_model.safetensors")
    ):
        raise ValueError("saved best adapter hash does not match its metadata")

    preflight.update(
        {
            "artifact_output": str(OUTPUT),
            "resume_from": str(checkpoint),
            "initialization": "exact continuation from the verified step-256 snapshot",
            "new_seed_or_hyperparameter_run": False,
            "resume_global_step": state["global_step"],
            "resume_sample_index": state["sample_index"],
            "resume_experiment_id": preflight["experiment_id"],
            "resume_training_experiment_id": training_experiment_id,
            "resume_consumed_id_order_sha256": state["consumed_sample_id_order_sha256"],
            "effective_config_sha256": effective_hash,
            "resume_snapshot_files": _checkpoint_file_hashes(checkpoint),
            "exact_resume_checks": {
                "config_corpus_manifest_and_code_hashes": "PASS",
                "effective_config_hash": "PASS",
                "optimizer_scheduler_adapter_and_rng_presence": "PASS",
                "sampler_order_and_consumed_counts": "PASS",
                "selector_best_adapter_hash": "PASS",
            },
            "parent_attempt_status": str(ATTEMPT_STATUS),
            "sealed_audit_loaded": False,
        }
    )
    return preflight, state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue the stopped Physion++ teacher run from its exact step-256 snapshot."
    )
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--e-long-adapter", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    preflight, _ = _preflight(args.reference_adapter, args.e_long_adapter)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("effective-config.json", preflight["config"]),
        ("run-invocation.json", preflight),
        (
            "resume-preflight.json",
            {
                "checks": preflight["exact_resume_checks"],
                "resume_from": str(RESUME_FROM),
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
                "ids": [
                    row.id
                    for row in original.deterministic_validation_subset(
                        original._read(original.CORPUS / "validation.jsonl"),
                        seed=17,
                        limit=2048,
                        video_task_weights=decision_training_config(
                            preflight["config"]
                        ).video_task_weights,
                    )
                ],
            },
        ),
    ):
        payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
        (OUTPUT / name).write_text(payload, encoding="utf-8")

    started = time.monotonic()
    result = original.run_training(
        train_path=original.CORPUS / "train.jsonl",
        validation_path=original.CORPUS / "validation.jsonl",
        config_path=original.CONFIG,
        output_dir=OUTPUT,
        reference_adapter_path=args.reference_adapter.resolve(),
        model_manifest_path=original.BASE_MANIFEST,
        media_root=original.HARDLINK_MEDIA_ROOT,
        resume_from=RESUME_FROM,
    )
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
