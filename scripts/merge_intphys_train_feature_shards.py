"""Merge pinned IntPhys train-shard caches and split identities globally."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audit_videoprism_checkpoint import CHECKPOINT_SHA256, MODEL_REVISION
from scripts.extract_frozen_videoprism_features import SOURCE_COMMIT
from scripts.extract_intphys_train_features import (
    TRAIN_ARCHIVE_PINS,
    deterministic_group_split,
    sha256_file,
)

IDENTITY_COLUMNS = (
    "status_identity_sha256",
    "rgb_sequence_sha256",
    "sampled_rgb_pixels_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        required = {"features", "sample_ids", "group_ids", *IDENTITY_COLUMNS}
        missing = required - set(cache.files)
        if missing:
            raise ValueError(f"feature cache {path} lacks columns: {sorted(missing)}")
        data = {name: cache[name] for name in required}
    count = len(data["sample_ids"])
    if count == 0:
        raise ValueError(f"feature cache {path} is empty")
    if data["features"].ndim != 3 or not np.isfinite(data["features"]).all():
        raise ValueError(f"feature cache {path} has invalid or non-finite features")
    if data["features"].shape[1:] != (8, 768):
        raise ValueError(f"feature cache {path} must contain [N, 8, 768] features")
    for name in ("sample_ids", "group_ids", *IDENTITY_COLUMNS):
        values = data[name].astype(str)
        if values.shape != (count,) or any(not value for value in values):
            raise ValueError(f"feature cache {path} has invalid {name}")
        data[name] = values
    for name in IDENTITY_COLUMNS:
        if any(_SHA256.fullmatch(value) is None for value in data[name]):
            raise ValueError(f"feature cache {path} has malformed {name}")
    if len(set(data["sample_ids"].tolist())) != count:
        raise ValueError(f"feature cache {path} contains duplicate sample IDs")
    return data


def _load_shard(feature_dir: Path) -> tuple[int, dict[str, Any], dict[str, np.ndarray]]:
    report_path = feature_dir / "extract-report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing shard extraction receipt: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete_intphys_train_frozen_feature_extraction":
        raise ValueError(
            f"shard receipt is not a completed IntPhys train extraction: {report_path}"
        )
    matching = [
        number
        for number, pin in TRAIN_ARCHIVE_PINS.items()
        if report.get("archive_source") == pin["url"]
        and report.get("archive_bytes") == pin["bytes"]
        and report.get("archive_md5") == pin["md5"]
    ]
    if len(matching) != 1:
        raise ValueError(f"shard receipt does not match one official archive pin: {report_path}")
    shard_number = matching[0]
    if report.get("archive_shard_number", shard_number) != shard_number:
        raise ValueError(f"shard receipt number conflicts with its pinned source: {report_path}")
    frame_count = report.get("frame_count")
    if frame_count is None:
        feature_shape = report.get("feature_shape")
        selected_frames = report.get("selected_frame_numbers_1_based")
        frame_count = feature_shape[0] if isinstance(feature_shape, list) else None
        if frame_count is None and isinstance(selected_frames, list):
            frame_count = len(selected_frames)
    if frame_count != 8:
        raise ValueError(f"all merged shards must use the same 8-frame policy: {report_path}")
    if report.get("model_revision") != MODEL_REVISION:
        raise ValueError(f"shard uses a different VideoPrism revision: {report_path}")
    if report.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError(f"shard uses a different VideoPrism checkpoint: {report_path}")
    if report.get("videoprism_source_commit") != SOURCE_COMMIT:
        raise ValueError(f"shard uses a different VideoPrism source commit: {report_path}")
    if not isinstance(report.get("archive_sha256"), str) or _SHA256.fullmatch(
        report["archive_sha256"]
    ) is None:
        raise ValueError(f"shard receipt lacks a source archive SHA-256: {report_path}")
    if report.get("all_train_labels_possible") is not True:
        raise ValueError(f"shard is not the official possible-only train split: {report_path}")

    split_files = report.get("feature_archive_sha256", {})
    chunks: list[dict[str, np.ndarray]] = []
    for split_name in ("train", "validation"):
        path = feature_dir / f"{split_name}-features.npz"
        if not path.is_file() or split_files.get(split_name) != sha256_file(path):
            raise ValueError(f"{split_name} cache does not match the shard receipt: {path}")
        chunks.append(_read_npz(path))
    if chunks[0]["features"].shape[1:] != chunks[1]["features"].shape[1:]:
        raise ValueError(f"train and validation feature shapes differ in {feature_dir}")
    if set(chunks[0]["sample_ids"]) & set(chunks[1]["sample_ids"]):
        raise ValueError(f"sample IDs cross the shard-local split in {feature_dir}")
    observed_count = sum(len(chunk["sample_ids"]) for chunk in chunks)
    if report.get("archive_videos") != observed_count:
        raise ValueError(f"feature count does not match source archive record count: {report_path}")

    combined = {
        name: np.concatenate([chunk[name] for chunk in chunks], axis=0)
        for name in ("features", "sample_ids", *IDENTITY_COLUMNS)
    }
    sample_ids = combined["sample_ids"].tolist()
    combined["sample_ids"] = np.asarray(
        [f"train.{shard_number}/{sample_id}" for sample_id in sample_ids], dtype=str
    )
    combined["shard_numbers"] = np.full(len(sample_ids), shard_number, dtype=np.int16)
    return shard_number, report, combined


def merge_feature_shards(
    feature_dirs: list[Path],
    output_dir: Path,
    *,
    seed: int = 17,
    required_shards: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, Any]:
    """Globally deduplicate identities and create one deterministic train/val split."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    loaded = []
    for path in feature_dirs:
        feature_dir = path.resolve(strict=True)
        loaded.append((feature_dir, *_load_shard(feature_dir)))
    shard_numbers = [number for _, number, _, _ in loaded]
    if len(set(shard_numbers)) != len(shard_numbers):
        raise ValueError("each official IntPhys train shard may appear only once")
    if set(shard_numbers) != set(required_shards):
        raise ValueError(
            f"expected exactly official shards {sorted(required_shards)}, "
            f"received {sorted(shard_numbers)}"
        )
    frame_shapes = {tuple(item[3]["features"].shape[1:]) for item in loaded}
    if len(frame_shapes) != 1:
        raise ValueError("feature sequence shapes differ between shards")

    ordered = sorted(loaded, key=lambda item: item[1])
    sample_ids = np.concatenate([item[3]["sample_ids"] for item in ordered])
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("qualified sample IDs are not unique")
    features = np.concatenate([item[3]["features"] for item in ordered], axis=0)
    status_hashes = np.concatenate([item[3]["status_identity_sha256"] for item in ordered])
    media_hashes = np.concatenate([item[3]["rgb_sequence_sha256"] for item in ordered])
    sampled_hashes = np.concatenate(
        [item[3]["sampled_rgb_pixels_sha256"] for item in ordered]
    )
    shard_numbers_per_sample = np.concatenate(
        [item[3]["shard_numbers"] for item in ordered]
    )

    train_indices, validation_indices, group_ids = deterministic_group_split(
        sample_ids.tolist(),
        status_hashes.tolist(),
        media_hashes.tolist(),
        sampled_media_hashes=sampled_hashes.tolist(),
        seed=seed,
    )
    train_groups = {group_ids[index] for index in train_indices}
    validation_groups = {group_ids[index] for index in validation_indices}
    train_identity_sets = [
        set(values[train_indices].tolist())
        for values in (status_hashes, media_hashes, sampled_hashes)
    ]
    validation_identity_sets = [
        set(values[validation_indices].tolist())
        for values in (status_hashes, media_hashes, sampled_hashes)
    ]
    if train_groups & validation_groups or any(
        left & right
        for left, right in zip(train_identity_sets, validation_identity_sets, strict=True)
    ):
        raise RuntimeError("global split failed an underlying-identity isolation check")

    output_dir.mkdir(parents=True, exist_ok=False)
    split_hashes: dict[str, str] = {}
    split_file_hashes: dict[str, str] = {}
    split_indices = {"train": train_indices, "validation": validation_indices}
    for split_name, indices in split_indices.items():
        ids = sample_ids[indices]
        split_hashes[split_name] = hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest()
        path = output_dir / f"{split_name}-features.npz"
        np.savez_compressed(
            path,
            features=features[indices],
            sample_ids=ids,
            source_sample_ids=np.asarray(
                [value.split("/", maxsplit=1)[1] for value in ids], dtype=str
            ),
            shard_numbers=shard_numbers_per_sample[indices],
            group_ids=np.asarray([group_ids[index] for index in indices], dtype=str),
            status_identity_sha256=status_hashes[indices],
            rgb_sequence_sha256=media_hashes[indices],
            sampled_rgb_pixels_sha256=sampled_hashes[indices],
        )
        split_file_hashes[split_name] = sha256_file(path)

    per_shard = []
    for feature_dir, shard_number, report, cache in ordered:
        per_shard.append({
            "shard_number": shard_number,
            "archive_url": report["archive_source"],
            "archive_bytes": report["archive_bytes"],
            "archive_md5": report["archive_md5"],
            "archive_sha256": report["archive_sha256"],
            "examples": int(len(cache["sample_ids"])),
            "feature_dir": str(feature_dir),
        })
    result = {
        "schema_version": 1,
        "status": "complete_intphys_all_train_shards_globally_grouped",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "required_shards": list(required_shards),
        "source_shards": per_shard,
        "model_revision": MODEL_REVISION,
        "videoprism_source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "frame_count": 8,
        "feature_shape_per_sample": list(next(iter(frame_shapes))),
        "sample_count": int(len(sample_ids)),
        "unique_identity_group_count": int(len(set(group_ids))),
        "cross_shard_duplicate_records_grouped": int(len(sample_ids) - len(set(group_ids))),
        "train_examples": int(len(train_indices)),
        "validation_examples": int(len(validation_indices)),
        "train_unique_groups": int(len(train_groups)),
        "validation_unique_groups": int(len(validation_groups)),
        "cross_split_group_overlap": 0,
        "cross_split_status_identity_overlap": 0,
        "cross_split_rgb_sequence_identity_overlap": 0,
        "cross_split_sampled_rgb_input_overlap": 0,
        "sample_id_order_sha256": split_hashes,
        "feature_archive_sha256": split_file_hashes,
        "split_policy": (
            "global union of exact status, full RGB sequence, and normalized sampled-input "
            "identity; SHA-256 group bucket 0/5 held for validation"
        ),
        "sealed_test_accessed": False,
        "quality_scope": "official IntPhys possible-only training corpus; no task-quality claim",
    }
    (output_dir / "merge-report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _args()
    report = merge_feature_shards(args.feature_dir, args.output_dir, seed=args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
