"""Build a source-group-disjoint native AVATAR localization fold from TRAIN annotations only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.fit_avatar_decision_readout import sha256_file, write_json_atomic

SEED = 17
TRAIN_FRACTION = 0.8
SINGLE_SOUND = "Single-Sound"
_CLIP_SUFFIX = re.compile(r"^(?P<source>.+)_\d{5}$")


def source_group_for_clip(clip_id: str) -> str:
    match = _CLIP_SUFFIX.fullmatch(clip_id)
    if not match:
        raise ValueError(f"clip id lacks the expected five-digit segment suffix: {clip_id}")
    return match.group("source")


def partition_source_groups(groups: set[str], *, seed: int = SEED) -> tuple[set[str], set[str]]:
    if len(groups) < 10:
        raise ValueError("at least ten source groups are required")
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"avatar-native-localization-fold-v1:{seed}:{group}".encode()
        ).hexdigest(),
    )
    boundary = round(len(ranked) * TRAIN_FRACTION)
    fit, validation = set(ranked[:boundary]), set(ranked[boundary:])
    if not fit or not validation or fit & validation or fit | validation != groups:
        raise ValueError("invalid source-group partition")
    return fit, validation


def _annotation_record(path: Path, value: dict[str, Any]) -> dict[str, Any] | None:
    annotations = value.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"annotations must be a list: {path}")
    selected = [item for item in annotations if item.get("task") == SINGLE_SOUND]
    if not selected:
        return None

    clip_id = path.parent.name
    frame_number = int(value["frame_number"])
    fps = float(value["fps"])
    width, height = int(value["original_width"]), int(value["original_height"])
    if path.stem != f"{frame_number:05d}" or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"inconsistent frame metadata: {path}")
    boxes = []
    tolerance = 1e-6
    for annotation in selected:
        bbox = [float(component) for component in annotation["bbox"]]
        if len(bbox) != 4 or not all(math.isfinite(number) for number in bbox):
            raise ValueError(f"invalid bounding box: {path}")
        x, y, box_width, box_height = bbox
        if (
            box_width <= 0
            or box_height <= 0
            or x < -tolerance
            or y < -tolerance
            or x + box_width > width + tolerance
            or y + box_height > height + tolerance
        ):
            raise ValueError(f"bounding box outside the source frame: {path}")
        x, y = max(0.0, x), max(0.0, y)
        boxes.append(
            [x, y, min(float(width), x + box_width) - x, min(float(height), y + box_height) - y]
        )

    source_group = source_group_for_clip(clip_id)
    return {
        "asset_key": f"{clip_id}/{frame_number:05d}",
        "clip_id": clip_id,
        "frame_number": frame_number,
        "timestamp_seconds": frame_number / fps,
        "source_group": source_group,
        "bbox_xywhs": boxes,
        "original_width": width,
        "original_height": height,
        "categories": sorted({str(item.get("audio_visual_category", "")) for item in selected}),
        "annotation_sha256": sha256_file(path),
    }


def build_fold(train_annotations: Path, output_dir: Path) -> dict[str, Any]:
    root = train_annotations.resolve(strict=True)
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    paths = sorted(root.rglob("*.json"))
    if not paths:
        raise ValueError("no TRAIN annotation JSON files found")

    records = []
    annotation_digests = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        record = _annotation_record(path, value)
        if record is not None:
            records.append(record)
            annotation_digests.append(record["annotation_sha256"])
    groups = {record["source_group"] for record in records}
    fit_groups, validation_groups = partition_source_groups(groups)
    splits = {
        "train": [record for record in records if record["source_group"] in fit_groups],
        "validation": [record for record in records if record["source_group"] in validation_groups],
    }
    if {row["source_group"] for row in splits["train"]} & {
        row["source_group"] for row in splits["validation"]
    }:
        raise ValueError("source-group leakage across splits")
    if {row["asset_key"] for row in splits["train"]} & {
        row["asset_key"] for row in splits["validation"]
    }:
        raise ValueError("media-identity leakage across splits")

    output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        destination = output / f"{split}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
    source_digest = hashlib.sha256("\n".join(sorted(annotation_digests)).encode()).hexdigest()
    report = {
        "experiment_id": "avatar-native-localization-train-only-fold-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_split": (
            "original AVATAR TRAIN annotations only; official validation and "
            "sealed audit not loaded"
        ),
        "seed": SEED,
        "partition": (
            "SHA-256 rank of source YouTube identity; 80% fit / 20% development validation"
        ),
        "source_train_annotation_file_count": len(paths),
        "single_sound_record_count": len(records),
        "source_group_count": len(groups),
        "source_annotation_digest_sha256": source_digest,
        "split_counts": {
            split: {
                "examples": len(rows),
                "source_groups": len({row["source_group"] for row in rows}),
                "clips": len({row["clip_id"] for row in rows}),
                "asset_keys_sha256": hashlib.sha256(
                    "\n".join(row["asset_key"] for row in rows).encode()
                ).hexdigest(),
                "jsonl_sha256": sha256_file(output / f"{split}.jsonl"),
            }
            for split, rows in splits.items()
        },
        "source_group_overlap": 0,
        "asset_key_overlap": 0,
        "sealed_audit_accessed": False,
    }
    write_json_atomic(output / "fold-manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_fold(args.train_annotations, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
