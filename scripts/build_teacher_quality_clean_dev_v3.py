from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.corpus import (  # noqa: E402
    filter_previously_seen_records,
    source_asset_identity,
)
from tiny_omni_decision.dataset import (  # noqa: E402
    audit_license,
    check_train_eval_splits,
    iter_local_rows,
    sha256_file,
)
from tiny_omni_decision.development import deterministic_asset_holdout  # noqa: E402
from tiny_omni_decision.io import load_structured_file  # noqa: E402
from tiny_omni_decision.schema import DecisionExample  # noqa: E402
from tiny_omni_decision.training import (  # noqa: E402
    decision_training_config,
    deterministic_sample_order,
)

BASE_REPO = Path(r"C:\Users\junny\OneDrive\ドキュメント\ChatGPT\tiny-omni-decision")
BASE_CORPUS = BASE_REPO / "data/processed/teacher-quality-next/clean-dev-v2"
BASE_CONFIG = BASE_REPO / "configs/decision/teacher_quality_clean_dev_v2.yaml"
V1_CORPUS = BASE_REPO / "data/processed/durable-teacher-v1"
PHYSION_CORPUS = Path(r"C:\CodexArtifacts\tqpp\fullclip-v1")
OUTPUT = Path(r"C:\CodexArtifacts\tqpp\physionpp-clean-dev-v3-generation-02\corpus")
MEDIA_ROOT = Path(r"C:\CodexArtifacts\tqpp\physionpp-augmented-v1\media-root")
SEED = 317
ASSETS_PER_MODALITY = 326


def _read(path: Path) -> list[DecisionExample]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [DecisionExample.model_validate(row) for row in iter_local_rows(path)]


def _id_hash(rows: list[DecisionExample]) -> str:
    return hashlib.sha256("".join(f"{row.id}\n" for row in rows).encode()).hexdigest()


def _counts(rows: list[DecisionExample]) -> dict[str, Any]:
    assets: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        assets.setdefault(row.modality, set()).add((row.source, source_asset_identity(row)))
    return {
        "examples": len(rows),
        "by_modality": dict(sorted(Counter(row.modality for row in rows).items())),
        "by_source": dict(sorted(Counter(row.source for row in rows).items())),
        "unique_asset_groups_by_modality": {
            modality: len(groups) for modality, groups in sorted(assets.items())
        },
        "unique_video_scenes": len(assets.get("video", set())),
        "unique_parent_questions": len(
            {(row.source, row.task_group_id or row.source_record_id) for row in rows}
        ),
        "task_types": {
            "unknown" if key is None else key: count
            for key, count in sorted(
                Counter(row.task_type for row in rows).items(),
                key=lambda item: "" if item[0] is None else item[0],
            )
        },
    }


def _write_rows(path: Path, rows: list[DecisionExample]) -> str:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json(exclude_none=True) + "\n")
    return sha256_file(path)


def _verify_media(rows: list[DecisionExample]) -> dict[str, str]:
    checked: dict[str, str] = {}
    for row in rows:
        for media in row.media:
            if not media.path:
                raise ValueError(f"unresolved media path: {row.id}")
            path = (MEDIA_ROOT / media.path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"media is missing for {row.id}: {path}")
            if media.sha256 and str(path) not in checked:
                actual = sha256_file(path)
                if actual != media.sha256:
                    raise ValueError(f"media digest mismatch for {row.id}: {path}")
                checked[str(path)] = actual
    return checked


def build() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite development corpus: {OUTPUT}")
    baseline_train = _read(BASE_CORPUS / "train.jsonl")
    baseline_validation = _read(BASE_CORPUS / "validation.jsonl")
    physion_train = _read(PHYSION_CORPUS / "train.jsonl")
    physion_validation = _read(PHYSION_CORPUS / "validation.jsonl")

    config = decision_training_config(load_structured_file(BASE_CONFIG))
    sampler_order, _ = deterministic_sample_order(
        baseline_train,
        seed=config.seed,
        limit=config.max_train_examples,
        modality_weights=config.modality_weights,
        source_weights=config.source_weights,
        max_sample_repeats=config.max_sample_repeats,
        video_task_weights=config.video_task_weights or None,
    )
    consumed = sampler_order[: 2048 * config.gradient_accumulation_steps]
    if len(consumed) != 8192 or len({row.id for row in consumed}) != 8192:
        raise ValueError("could not reproduce 8,192 unique examples consumed by E-long")
    consumed_hash = _id_hash(consumed)
    expected_consumed_hash = "ccd1a0299ccca88de9eef08f5263ac5b5ca6bff851fba8509c6cb99a9f1d9e24"
    if consumed_hash != expected_consumed_hash:
        raise ValueError(f"E-long consumed ID order changed: {consumed_hash}")

    # The source Teacher v1 corpus train and validation splits were both observed.
    prior_seen = [
        *consumed,
        *baseline_validation,
        *_read(V1_CORPUS / "train.jsonl"),
        *_read(V1_CORPUS / "validation.jsonl"),
        *physion_validation,
    ]
    candidates = [*baseline_train, *physion_train]
    unseen, exclusions = filter_previously_seen_records(candidates, prior_seen)

    selected: list[DecisionExample] = []
    held_groups: set[tuple[str, str]] = set()
    selection: dict[str, Any] = {}
    for modality in ("text", "image", "audio", "video"):
        eligible = [row for row in unseen if row.modality == modality]
        rows, groups = deterministic_asset_holdout(
            eligible, modality=modality, count=ASSETS_PER_MODALITY, seed=SEED
        )
        rows = [row.model_copy(update={"split": "validation"}) for row in rows]
        selected.extend(rows)
        held_groups.update(groups)
        selection[modality] = {
            "eligible_rows": len(eligible),
            "eligible_asset_groups": len(
                {(row.source, source_asset_identity(row)) for row in eligible}
            ),
            "selected_examples": len(rows),
            "selected_id_order_sha256": _id_hash(rows),
            "source_counts": dict(sorted(Counter(row.source for row in rows).items())),
        }

    train = [
        row.model_copy(update={"split": "train"})
        for row in candidates
        if (row.source, source_asset_identity(row)) not in held_groups
    ]
    integrity = check_train_eval_splits(train, selected)
    if integrity.get("status") != "disjoint":
        raise ValueError(f"clean-dev-v3 split integrity failed: {integrity}")
    for row in (*train, *selected):
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
            raise ValueError(f"non-ALLOW record {row.id}: {unresolved}")
    media_hashes = _verify_media(selected)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    train_hash = _write_rows(OUTPUT / "train.jsonl", train)
    validation_hash = _write_rows(OUTPUT / "validation.jsonl", selected)
    result = {
        "generation_id": "teacher-quality-clean-dev-v3-physionpp",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "selected_assets_per_modality": ASSETS_PER_MODALITY,
        "source_corpora_read_only": True,
        "input_sha256": {
            str(path): sha256_file(path)
            for path in (
                BASE_CORPUS / "train.jsonl",
                BASE_CORPUS / "validation.jsonl",
                BASE_CORPUS / "manifest.json",
                PHYSION_CORPUS / "train.jsonl",
                PHYSION_CORPUS / "validation.jsonl",
                PHYSION_CORPUS / "corpus-manifest.json",
                V1_CORPUS / "train.jsonl",
                V1_CORPUS / "validation.jsonl",
            )
        },
        "e_long_consumed_ids": 8192,
        "e_long_consumed_id_order_sha256": consumed_hash,
        "seen_record_filter": exclusions,
        "selected_by_modality": selection,
        "train": _counts(train),
        "validation": _counts(selected),
        "unique_validation_media_hashes_verified": len(media_hashes),
        "train_sha256": train_hash,
        "validation_sha256": validation_hash,
        "validation_id_order_sha256": _id_hash(selected),
        "train_validation_integrity": integrity,
        "sealed_audit_loaded": False,
        "training_run_corpus_ids_are_unconsumed": True,
        "notes": [
            "The selector uses asset groups absent from E-long consumed IDs, its full prior "
            "validation, Teacher v1 train/validation, and Physion++ prior validation.",
            "The candidate pool includes clean-dev-v2 source-train records not consumed by its "
            "8,192-example path; selected groups are removed from this run's train split.",
            "This is development validation and not a sealed audit.",
        ],
    }
    (OUTPUT / "corpus-manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
