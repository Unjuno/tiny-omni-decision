from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.corpus import source_asset_identity  # noqa: E402
from tiny_omni_decision.dataset import (  # noqa: E402
    audit_license,
    check_train_eval_splits,
    sha256_file,
)
from tiny_omni_decision.io import load_structured_file  # noqa: E402
from tiny_omni_decision.schema import BaseModelManifest, DecisionExample  # noqa: E402
from tiny_omni_decision.trainer import run_training  # noqa: E402
from tiny_omni_decision.training import (  # noqa: E402
    decision_training_config,
    deterministic_sample_order,
    deterministic_validation_subset,
    sampling_accounting,
)

CORPUS = Path(r"C:\CodexArtifacts\tqpp\physionpp-clean-dev-v3-generation-02\corpus")
MEDIA_ROOT = Path(r"C:\CodexArtifacts\tqpp\physionpp-augmented-v1\media-root")
HARDLINK_MEDIA_ROOT = Path(
    r"C:\CodexArtifacts\tqpp\physionpp-clean-dev-v3-generation-02\media-root"
)
OUTPUT = Path(
    r"C:\CodexArtifacts\tqpp\physionpp-clean-dev-v3-generation-02\run-seed17-2048-attempt-02"
)
CONFIG = ROOT / "configs/decision/teacher_v2_physionpp_clean_dev_v3.yaml"
CONTROL_CONFIG = ROOT / "configs/decision/teacher_quality_clean_dev_v2.yaml"
BASE_MANIFEST = ROOT / "manifests/base-model.example.yaml"
BRANCH = "codex/teacher-quality-physionpp"
PREFLIGHT_NAME = "run-preflight-hardlink-v1.json"


def _read(path: Path) -> list[DecisionExample]:
    with path.open(encoding="utf-8") as handle:
        return [DecisionExample.model_validate_json(line) for line in handle if line.strip()]


def _hash_ids(rows: list[DecisionExample]) -> str:
    return hashlib.sha256("".join(f"{row.id}\n" for row in rows).encode()).hexdigest()


def _counts(rows: list[DecisionExample]) -> dict[str, Any]:
    assets: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        assets.setdefault(row.modality, set()).add((row.source, source_asset_identity(row)))
    return {
        "examples": len(rows),
        "by_modality": dict(sorted(Counter(row.modality for row in rows).items())),
        "by_source": dict(sorted(Counter(row.source for row in rows).items())),
        "unique_assets_by_modality": {key: len(value) for key, value in sorted(assets.items())},
        "video_task_types": dict(
            sorted(Counter(row.task_type for row in rows if row.modality == "video").items())
        ),
    }


def _materialize_media_hardlinks(examples: list[DecisionExample]) -> dict[str, Any]:
    reuse = HARDLINK_MEDIA_ROOT.exists()
    if not reuse:
        HARDLINK_MEDIA_ROOT.mkdir(parents=True)
    linked: dict[str, str] = {}
    source_by_relative: dict[str, Path] = {}
    digest_by_relative: dict[str, str | None] = {}
    for row in examples:
        for media in row.media:
            if not media.path:
                raise ValueError(f"unresolved media path in example {row.id}")
            relative = Path(media.path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"media path is not a safe relative path: {media.path}")
            relative_key = relative.as_posix().casefold()
            source = (MEDIA_ROOT / relative).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"missing media for {row.id}: {source}")
            destination = HARDLINK_MEDIA_ROOT / relative
            if relative_key in source_by_relative:
                if source_by_relative[relative_key] != source:
                    raise ValueError(f"conflicting source paths for {media.path}")
                if media.sha256 and digest_by_relative[relative_key] not in {
                    None,
                    media.sha256,
                }:
                    raise ValueError(f"conflicting media digests for {media.path}")
                digest_by_relative[relative_key] = digest_by_relative[relative_key] or media.sha256
                continue
            source_by_relative[relative_key] = source
            digest_by_relative[relative_key] = media.sha256
            destination.parent.mkdir(parents=True, exist_ok=True)
            if reuse:
                if not destination.is_file():
                    raise FileNotFoundError(f"partial media overlay is missing {destination}")
            else:
                os.link(source, destination)
            if source.stat().st_dev != destination.stat().st_dev:
                raise RuntimeError("media hardlink is not on the same volume as its source")
            if source.stat().st_ino != destination.stat().st_ino:
                raise ValueError(f"media overlay entry is not a hardlink: {destination}")
            linked[media.path] = sha256_file(destination)
            if media.sha256 and linked[media.path] != media.sha256:
                raise ValueError(f"hardlinked media hash mismatch: {destination}")
    return {
        "root": str(HARDLINK_MEDIA_ROOT),
        "files": len(linked),
        "manifest_sha256": hashlib.sha256(
            "".join(f"{key}\0{linked[key]}\n" for key in sorted(linked)).encode()
        ).hexdigest(),
        "storage_method": "NTFS hardlinks; source media unchanged",
        "reused_verified_overlay": reuse,
    }


def _preflight(reference_adapter: Path, e_long_adapter: Path) -> dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != BRANCH:
        raise ValueError(f"run must use {BRANCH}; found {branch}")
    if not (reference_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Teacher v1 comparison adapter missing: {reference_adapter}")
    if not (e_long_adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"E-long selected adapter missing: {e_long_adapter}")
    train_path, validation_path = CORPUS / "train.jsonl", CORPUS / "validation.jsonl"
    manifest_path = CORPUS / "corpus-manifest.json"
    for path in (train_path, validation_path, manifest_path, MEDIA_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)

    candidate_config = load_structured_file(CONFIG)
    control_config = load_structured_file(CONTROL_CONFIG)
    candidate_as_control = json.loads(json.dumps(candidate_config))
    candidate_as_control["teacher_id"] = control_config["teacher_id"]
    if candidate_as_control != control_config:
        raise ValueError("candidate config must match clean-dev-v2 control except experiment ID")
    config = decision_training_config(candidate_config)
    required = (
        config.seed == 17
        and config.max_steps == 2048
        and config.gradient_accumulation_steps == 4
        and config.video_num_frames == 8
        and config.lora_target_policy == "decoder_all_linear"
        and config.lora_rank == 16
        and not config.use_rslora
        and config.max_sample_repeats == 1
        and config.early_stopping_patience == 17
    )
    if not required:
        raise ValueError("effective config differs from the frozen experiment conditions")

    corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corpus_manifest.get("sealed_audit_loaded") is not False:
        raise ValueError("clean-dev-v3 corpus manifest indicates sealed audit access")
    if corpus_manifest.get("train_sha256") != sha256_file(train_path):
        raise ValueError("train JSONL hash differs from clean-dev-v3 manifest")
    if corpus_manifest.get("validation_sha256") != sha256_file(validation_path):
        raise ValueError("validation JSONL hash differs from clean-dev-v3 manifest")
    if corpus_manifest.get("train_validation_integrity", {}).get("status") != "disjoint":
        raise ValueError("clean-dev-v3 corpus manifest is not disjoint")

    train, validation = _read(train_path), _read(validation_path)
    integrity = check_train_eval_splits(train, validation)
    if integrity.get("status") != "disjoint":
        raise ValueError(f"train/validation gate failed: {integrity}")
    for row in (*train, *validation):
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
            raise ValueError(f"non-ALLOW row {row.id}: {unresolved}")

    sample_order, _ = deterministic_sample_order(
        train,
        seed=config.seed,
        limit=config.max_train_examples,
        modality_weights=config.modality_weights,
        source_weights=config.source_weights,
        max_sample_repeats=config.max_sample_repeats,
        video_task_weights=config.video_task_weights or None,
    )
    consumed = sample_order[: config.max_steps * config.gradient_accumulation_steps]
    if len(consumed) != 8192 or len({item.id for item in consumed}) != 8192:
        raise ValueError("training sampler cannot supply 8,192 unique examples")
    selector = deterministic_validation_subset(
        validation,
        seed=config.seed,
        limit=config.selection_eval_examples,
        video_task_weights=config.video_task_weights or None,
    )
    selector_counts = Counter(item.modality for item in selector)
    if selector_counts != {modality: 326 for modality in ("text", "image", "audio", "video")}:
        raise ValueError(
            f"selector is not balanced at 326 examples per modality: {selector_counts}"
        )
    if len({item.id for item in selector}) != len(selector):
        raise ValueError("selector contains repeated IDs")

    media_checked: dict[str, str] = {}
    for row in (*consumed, *selector):
        for media in row.media:
            if not media.path:
                raise ValueError(f"unresolved media path in selected example {row.id}")
            media_path = (MEDIA_ROOT / media.path).resolve()
            if not media_path.is_file():
                raise FileNotFoundError(f"missing media for {row.id}: {media_path}")
            if media.sha256 and str(media_path) not in media_checked:
                digest = sha256_file(media_path)
                if digest != media.sha256:
                    raise ValueError(f"media hash mismatch for {row.id}: {media_path}")
                media_checked[str(media_path)] = digest

    media_overlay = _materialize_media_hardlinks([*consumed, *selector])

    base_manifest = BaseModelManifest.model_validate(load_structured_file(BASE_MANIFEST))
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_id": "teacher-v2-physionpp-clean-dev-v3-seed17-2048",
        "branch": branch,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "initialization": (
            "fresh pinned pretrained base and newly initialized LoRA; no warm-start adapter"
        ),
        "resume_from": None,
        "reference_adapters": {
            "teacher_v1": str(reference_adapter),
            "e_long_selected": str(e_long_adapter),
            "usage": "read-only evaluation only",
        },
        "sealed_audit_loaded": False,
        "config": candidate_config,
        "config_sha256": sha256_file(CONFIG),
        "control_config_sha256": sha256_file(CONTROL_CONFIG),
        "base_manifest_sha256": sha256_file(BASE_MANIFEST),
        "base_revision": base_manifest.revision,
        "processor_revision": base_manifest.processor_revision,
        "base_weights_sha256": next(
            (
                item["sha256"]
                for item in base_manifest.files
                if item.get("path") == "model.safetensors"
            ),
            None,
        ),
        "corpus_manifest_sha256": sha256_file(manifest_path),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_validation_integrity": integrity,
        "sampler_order_id_sha256": _hash_ids(sample_order),
        "planned_consumption_id_order_sha256": _hash_ids(consumed),
        "planned_consumption": sampling_accounting(consumed),
        "planned_optimizer_steps": config.max_steps,
        "planned_unique_examples_consumed": len(consumed),
        "validation_selector_id_order_sha256": _hash_ids(selector),
        "validation_selector_counts": _counts(selector),
        "media_hashes_verified": len(media_checked),
        "schedule": {
            "scheduler": config.lr_scheduler,
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "total_steps": config.max_steps,
            "patience": config.early_stopping_patience,
            "evaluation_interval": config.evaluation_interval,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "torchvision", "av")
            },
            "gpu": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
            "attention_backend": "PyTorch math SDP; flash and memory-efficient SDP disabled",
            "disk_free_bytes": shutil.disk_usage(MEDIA_ROOT).free,
        },
        "media_root": str(HARDLINK_MEDIA_ROOT),
        "media_source_root": str(MEDIA_ROOT),
        "media_overlay": media_overlay,
        "artifact_output": str(OUTPUT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fresh 2,048-step clean-dev-v3 Teacher candidate."
    )
    parser.add_argument("--reference-adapter", type=Path, required=True)
    parser.add_argument("--e-long-adapter", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {OUTPUT}")
    preflight = _preflight(args.reference_adapter.resolve(), args.e_long_adapter.resolve())
    if args.preflight_only:
        preflight_path = CORPUS / PREFLIGHT_NAME
        if preflight_path.exists():
            raise FileExistsError(f"refusing to overwrite saved preflight: {preflight_path}")
        preflight_path.write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for filename, value in (
        ("effective-config.json", preflight["config"]),
        ("run-invocation.json", preflight),
        (
            "validation-selector.json",
            {
                "corpus_sha256": preflight["validation_sha256"],
                "id_order_sha256": preflight["validation_selector_id_order_sha256"],
                "ids": [
                    row.id
                    for row in deterministic_validation_subset(
                        _read(CORPUS / "validation.jsonl"),
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
        (OUTPUT / filename).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    started = time.monotonic()
    result = run_training(
        train_path=CORPUS / "train.jsonl",
        validation_path=CORPUS / "validation.jsonl",
        config_path=CONFIG,
        output_dir=OUTPUT,
        reference_adapter_path=args.reference_adapter.resolve(),
        model_manifest_path=BASE_MANIFEST,
        media_root=HARDLINK_MEDIA_ROOT,
        resume_from=None,
    )
    summary = {
        "experiment_id": preflight["experiment_id"],
        "status": "completed_training" if result["global_steps"] == 2048 else "incomplete",
        "global_steps": result["global_steps"],
        "best_checkpoint_step": result["best_checkpoint_step"],
        "validation_best_metrics": result["validation_best_metrics"],
        "validation_reference_metrics": result.get("validation_reference_metrics"),
        "checkpoint_reload_verified": result["checkpoint_reload_verified"],
        "training_seconds": result["training_seconds"],
        "wall_seconds": time.monotonic() - started,
        "peak_vram_bytes": result["max_allocated_vram_bytes"],
        "selected_adapter_sha256": result.get("selected_adapter_sha256"),
        "train_corpus_sha256": preflight["train_sha256"],
        "validation_corpus_sha256": preflight["validation_sha256"],
    }
    (OUTPUT / "run-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
