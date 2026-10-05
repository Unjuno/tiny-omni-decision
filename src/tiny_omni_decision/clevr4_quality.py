from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from .dataset import CLEVR4_TAXONOMIES, _flatten_clevr4
from .schema import DatasetManifest, DecisionExample


def _bucket(asset_id: str, seed: int) -> int:
    payload = f"teacher-quality-next-clevr4\0{seed}\0{asset_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 100


def build_clevr4_quality_splits(
    annotations: dict[str, dict[str, Any]],
    *,
    previously_seen_image_ids: set[str],
    manifest: DatasetManifest,
    seed: int = 17,
) -> tuple[list[DecisionExample], list[DecisionExample], dict[str, Any]]:
    """Split only fresh official-train images; retain an unmaterialized audit reserve."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    training: list[DecisionExample] = []
    validation: list[DecisionExample] = []
    reserve: list[str] = []
    used_ids: set[str] = set()
    counts: Counter[str] = Counter()
    per_taxonomy: dict[str, dict[str, Counter[str]]] = {
        split: {taxonomy: Counter() for taxonomy in CLEVR4_TAXONOMIES}
        for split in ("train", "validation")
    }
    for asset_id, row in sorted(annotations.items()):
        if row.get("split") != "train" or asset_id in previously_seen_image_ids:
            continue
        used_ids.add(asset_id)
        bucket = _bucket(asset_id, seed)
        if bucket >= 85:
            reserve.append(asset_id)
            continue
        split = "train" if bucket < 70 else "validation"
        target = training if split == "train" else validation
        counts[f"{split}_images"] += 1
        converted = _flatten_clevr4({**row, "id": asset_id}, manifest)
        for example in converted:
            taxonomy = example.id.rsplit(":", 1)[-1]
            per_taxonomy[split][taxonomy][example.target] += 1
            target.append(
                example.model_copy(
                    update={"split": split, "task_group_id": f"image:{asset_id}"}
                )
            )
    if not training or not validation or not reserve:
        raise ValueError("fresh Clevr-4 assets must populate train, validation, and reserve")
    train_assets = {item.task_group_id for item in training}
    validation_assets = {item.task_group_id for item in validation}
    reserve_assets = {f"image:{item}" for item in reserve}
    if (
        train_assets & validation_assets
        or train_assets & reserve_assets
        or validation_assets & reserve_assets
    ):
        raise AssertionError("Clevr-4 image identity leaked across partitions")
    for taxonomy, vocabulary in CLEVR4_TAXONOMIES.items():
        missing = set(map(str, vocabulary)) - set(per_taxonomy["validation"][taxonomy])
        if missing:
            raise ValueError(f"validation lacks Clevr-4 {taxonomy} classes: {sorted(missing)}")

    reserved_id_hash = hashlib.sha256("\n".join(sorted(reserve)).encode()).hexdigest()
    metadata: dict[str, Any] = {
        "seed": seed,
        "split_percentages": {"train": 70, "validation": 15, "unmaterialized_reserve": 15},
        "fresh_official_train_images": len(used_ids),
        "train_unique_images": counts["train_images"],
        "validation_unique_images": counts["validation_images"],
        "unmaterialized_reserve_images": len(reserve),
        "reserve_identity_sha256": reserved_id_hash,
        "train_examples": len(training),
        "validation_examples": len(validation),
        "train_validation_image_overlap": 0,
        "class_counts_by_taxonomy": {
            split: {
                taxonomy: dict(sorted(counter.items()))
                for taxonomy, counter in taxonomies.items()
            }
            for split, taxonomies in per_taxonomy.items()
        },
    }
    return training, validation, metadata
