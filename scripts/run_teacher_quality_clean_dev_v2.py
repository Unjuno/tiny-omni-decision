from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.corpus import (  # noqa: E402
    source_asset_identity,
    validate_training_inputs,
)
from tiny_omni_decision.dataset import (  # noqa: E402
    audit_license,
    check_train_eval_splits,
    iter_local_rows,
    sha256_file,
)
from tiny_omni_decision.io import load_structured_file  # noqa: E402
from tiny_omni_decision.schema import DecisionExample  # noqa: E402
from tiny_omni_decision.trainer import run_training  # noqa: E402
from tiny_omni_decision.training import (  # noqa: E402
    decision_training_config,
    deterministic_sample_order,
    deterministic_validation_subset,
)

TRAIN = ROOT / "data/processed/teacher-quality-next/clean-dev-v2/train.jsonl"
VALIDATION = ROOT / "data/processed/teacher-quality-next/clean-dev-v2/validation.jsonl"
CORPUS_MANIFEST = ROOT / "data/processed/teacher-quality-next/clean-dev-v2/manifest.json"
CONFIG = ROOT / "configs/decision/teacher_quality_clean_dev_v2.yaml"
BASE_MANIFEST = ROOT / "manifests/base-model.example.yaml"
REFERENCE_ADAPTER = ROOT / "artifacts/tiny-omni-decision-teacher-v1/selected"
OUTPUT = ROOT / "artifacts/tiny-omni-decision-teacher-v2/clean-dev-v2-seed17-2048"
BRANCH = "codex/teacher-quality-clean-dev-v1"


def run() -> dict[str, object]:
    current_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if current_branch != BRANCH:
        raise ValueError(f"run must execute from {BRANCH}; found {current_branch}")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite an existing experiment: {OUTPUT}")
    if not all(path.is_file() for path in (TRAIN, VALIDATION, CORPUS_MANIFEST, CONFIG)):
        raise FileNotFoundError("clean corpus, manifest, or config is missing")
    if not (REFERENCE_ADAPTER / "adapter_config.json").is_file():
        raise FileNotFoundError("read-only Teacher v1 reference adapter is missing")

    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("sealed_audit_loaded") or manifest.get("test_partitions_loaded"):
        raise ValueError("clean development manifest records forbidden audit/test access")
    if manifest.get("train_validation_integrity", {}).get("status") != "disjoint":
        raise ValueError("clean development corpus failed its split integrity gate")
    if manifest.get("train_sha256") != sha256_file(TRAIN):
        raise ValueError("training corpus hash differs from its frozen manifest")
    if manifest.get("validation_sha256") != sha256_file(VALIDATION):
        raise ValueError("validation corpus hash differs from its frozen manifest")
    validate_training_inputs(TRAIN, VALIDATION)

    config = load_structured_file(CONFIG)
    if config["training"]["seed"] != 17 or config["training"]["max_steps"] != 2048:
        raise ValueError("seed or planned update budget differs from the frozen experiment")
    if config["training"]["selection_eval_examples"] != 2048:
        raise ValueError("expected a fixed, balanced 512-example-per-modality selector set")
    if config["training"]["lora_target_policy"] != "decoder_all_linear":
        raise ValueError("candidate target policy differs from the frozen experiment")
    base_manifest = load_structured_file(BASE_MANIFEST)
    training_config = decision_training_config(config)
    train_rows = [DecisionExample.model_validate(row) for row in iter_local_rows(TRAIN)]
    validation_rows = [
        DecisionExample.model_validate(row) for row in iter_local_rows(VALIDATION)
    ]
    integrity = check_train_eval_splits(train_rows, validation_rows)
    if integrity["status"] != "disjoint":
        raise ValueError(f"train/validation integrity gate failed: {integrity}")
    for row in [*train_rows, *validation_rows]:
        policy, unresolved = audit_license(
            row.provenance.license,
            commercial_use=row.provenance.commercial_use,
            derivative_model_training_allowed=row.provenance.derivative_model_training_allowed,
            redistribution_allowed=row.provenance.redistribution_allowed,
            media_redistribution_allowed=row.provenance.media_redistribution_allowed,
            has_media=bool(row.media),
            trust_status=row.provenance.trust_status,
        )
        if policy != "ALLOW":
            raise ValueError(f"corpus contains a non-ALLOW example: {row.id}: {unresolved}")
    order, _ = deterministic_sample_order(
        train_rows,
        seed=training_config.seed,
        limit=training_config.max_train_examples,
        modality_weights=training_config.modality_weights,
        source_weights=training_config.source_weights,
        max_sample_repeats=training_config.max_sample_repeats,
        video_task_weights=training_config.video_task_weights or None,
    )
    run_samples = training_config.max_steps * training_config.gradient_accumulation_steps
    if len(order) < run_samples:
        raise ValueError("sampler cannot supply one unique example per planned microbatch")
    if len({row.id for row in order}) != len(order):
        raise ValueError("sampler preflight contains repeated sample IDs")

    def sample_counts(rows: list[DecisionExample]) -> dict[str, object]:
        modality = Counter(row.modality for row in rows)
        sources = Counter(row.source for row in rows)
        source_modality = Counter(f"{row.source}:{row.modality}" for row in rows)
        assets: dict[str, set[str]] = {}
        for row in rows:
            assets.setdefault(row.modality, set()).add(
                f"{row.source}:{source_asset_identity(row)}"
            )
        return {
            "examples": len(rows),
            "by_modality": dict(sorted(modality.items())),
            "by_source": dict(sorted(sources.items())),
            "by_source_modality": dict(sorted(source_modality.items())),
            "unique_assets_by_modality": {
                name: len(values) for name, values in sorted(assets.items())
            },
            "video_task_types": dict(
                sorted(Counter(row.task_type for row in rows if row.modality == "video").items())
            ),
            "unique_video_scenes": len(
                {
                    source_asset_identity(row)
                    for row in rows
                    if row.modality == "video"
                }
            ),
        }

    prefix = order[:run_samples]
    validation_selector = deterministic_validation_subset(
        validation_rows,
        seed=training_config.seed,
        limit=training_config.selection_eval_examples,
        video_task_weights=training_config.video_task_weights or None,
    )
    sampler_preflight = {
        "seed": training_config.seed,
        "configured_unique_order_examples": len(order),
        "planned_optimizer_updates": training_config.max_steps,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "planned_unique_microbatch_examples": run_samples,
        "max_sample_repeats": training_config.max_sample_repeats,
        "full_order_sha256": hashlib.sha256(
            "".join(f"{row.id}\n" for row in order).encode()
        ).hexdigest(),
        "planned_consumed_id_order_sha256": hashlib.sha256(
            "".join(f"{row.id}\n" for row in prefix).encode()
        ).hexdigest(),
        "configured_order": sample_counts(order),
        "planned_run_consumption": sample_counts(prefix),
        "validation_selector_examples": len(validation_selector),
        "validation_selector_id_order_sha256": hashlib.sha256(
            "".join(f"{row.id}\n" for row in validation_selector).encode()
        ).hexdigest(),
        "validation_selector_by_modality": dict(
            sorted(Counter(row.modality for row in validation_selector).items())
        ),
        "sealed_audit_loaded": False,
    }
    packages = {}
    for name in ("torch", "transformers", "peft", "torchvision", "av", "torchcodec"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu = "UNKNOWN"
    invocation = {
        "experiment_id": "teacher-quality-clean-dev-v2-seed17-2048",
        "branch": current_branch,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "train_sha256": sha256_file(TRAIN),
        "validation_sha256": sha256_file(VALIDATION),
        "corpus_manifest_sha256": sha256_file(CORPUS_MANIFEST),
        "config_sha256": sha256_file(CONFIG),
        "base_manifest_sha256": sha256_file(BASE_MANIFEST),
        "reference_adapter_sha256": sha256_file(
            REFERENCE_ADAPTER / "adapter_model.safetensors"
        ),
        "initialization": "fresh pinned pretrained base + newly initialized LoRA",
        "reference_adapter_role": "read-only comparison metadata only",
        "resume_from": None,
        "sealed_audit_loaded": False,
        "hyperparameter_hypothesis": "additional unique multimodal training coverage",
        "architecture_unchanged_from_E_long": True,
        "source_count_training": manifest["train"],
        "source_count_validation": manifest["validation"],
        "base_revision": base_manifest.get("revision"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "gpu": gpu,
            "attention_backend": "PyTorch math SDP; flash and memory-efficient SDP disabled",
            "windows_loader": "bounded-memory safetensors loader patch in trainer.py",
        },
        "config": config,
    }
    invocation["invocation_sha256"] = hashlib.sha256(
        json.dumps(invocation, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "run-invocation.json").write_text(
        json.dumps(invocation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "sampler-preflight.json").write_text(
        json.dumps(sampler_preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUTPUT / "validation-selector.json").write_text(
        json.dumps(
            {
                "corpus_sha256": sha256_file(VALIDATION),
                "seed": training_config.seed,
                "selection_eval_examples": training_config.selection_eval_examples,
                "id_order_sha256": sampler_preflight[
                    "validation_selector_id_order_sha256"
                ],
                "ids": [row.id for row in validation_selector],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = run_training(
        train_path=TRAIN,
        validation_path=VALIDATION,
        config_path=CONFIG,
        output_dir=OUTPUT,
        reference_adapter_path=REFERENCE_ADAPTER,
        model_manifest_path=BASE_MANIFEST,
        media_root=ROOT / "data",
        resume_from=None,
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
    (OUTPUT / "run-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
