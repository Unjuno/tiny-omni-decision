"""Measure label-free temporal variation in pinned Physical Concepts train/validation shards."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.audit_physical_concepts_tfrecord import _feature_map, iter_tfrecord_examples


def _file_receipt(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return size, base64.b64encode(md5.digest()).decode("ascii"), sha256.hexdigest()


def sequence_signal(
    image_bytes: bytes,
    mask_bytes: bytes,
    pose_values: list[float],
    np: Any,
) -> dict[str, Any]:
    """Summarize motion, segmentation change, and camera-state variation per sequence."""
    if len(image_bytes) != 15 * 64 * 64 * 3 or len(mask_bytes) != 15 * 64 * 64:
        raise ValueError("unexpected Physical Concepts image/mask shape")
    if len(pose_values) != 90:
        raise ValueError("camera pose must have 15x6 values")
    rgb = np.frombuffer(image_bytes, dtype=np.uint8).reshape(15, 64, 64, 3)
    mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(15, 64, 64)
    pose = np.asarray(pose_values, dtype=np.float32).reshape(15, 6)
    rgb_delta = np.abs(np.diff(rgb.astype(np.int16), axis=0)).mean(axis=(1, 2, 3)) / 255.0
    mask_delta = np.mean(mask[1:] != mask[:-1], axis=(1, 2))
    pose_delta = np.linalg.norm(np.diff(pose, axis=0), axis=1)
    if (
        not np.isfinite(rgb_delta).all()
        or not np.isfinite(mask_delta).all()
        or not np.isfinite(pose_delta).all()
    ):
        raise ValueError("non-finite temporal signal")
    unique_rgb_frames = len(
        {hashlib.sha256(rgb[index].tobytes()).digest() for index in range(15)}
    )
    unique_mask_frames = len(
        {hashlib.sha256(mask[index].tobytes()).digest() for index in range(15)}
    )
    unique_pose_rows = len({pose[index].tobytes() for index in range(15)})
    return {
        "rgb_adjacent_mae_mean": float(rgb_delta.mean()),
        "rgb_adjacent_mae_max": float(rgb_delta.max()),
        "rgb_identical_transition_count": int((rgb_delta == 0).sum()),
        "unique_rgb_frames": unique_rgb_frames,
        "mask_changed_pixel_fraction_mean": float(mask_delta.mean()),
        "mask_changed_transition_count": int((mask_delta > 0).sum()),
        "unique_mask_frames": unique_mask_frames,
        "mask_unique_value_count": int(np.unique(mask).size),
        "mask_foreground_fraction": float((mask > 0).mean()),
        "camera_pose_delta_l2_mean": float(pose_delta.mean()),
        "camera_pose_static_transition_count": int((pose_delta == 0).sum()),
        "unique_camera_pose_rows": unique_pose_rows,
    }


def _percentiles(values: Any, np: Any) -> dict[str, float]:
    result = np.percentile(values, [0, 10, 50, 90, 100])
    return {
        key: float(value)
        for key, value in zip(("min", "p10", "p50", "p90", "max"), result, strict=True)
    }


def _load_split(
    entry: dict[str, Any], report: dict[str, Any], np: Any
) -> tuple[int, dict[str, Any]]:
    if (
        report.get("status") != "complete"
        or report.get("object_name") != entry.get("object_name")
        or report.get("generation") != entry.get("generation")
    ):
        raise ValueError("audit report does not match the pinned split entry")
    path = Path(entry["local_payload_path"]).resolve(strict=True)
    size, md5_b64, sha256 = _file_receipt(path)
    if (
        size != entry["compressed_bytes"]
        or md5_b64 != entry["md5_base64"]
        or sha256 != entry["sha256"]
        or sha256 != report.get("source_sha256")
    ):
        raise ValueError("source shard fails pinned file receipt")

    expected_rows = report["record_sha256"]
    metrics: list[dict[str, Any]] = []
    for index, serialized in enumerate(iter_tfrecord_examples(path)):
        if (
            index >= len(expected_rows)
            or hashlib.sha256(serialized).hexdigest() != expected_rows[index]
        ):
            raise ValueError(f"row hash mismatch at index {index}")
        features = _feature_map(serialized)
        if set(features) != {"image", "mask", "camera_pose"}:
            raise ValueError(f"unexpected feature names at row {index}")
        image_kind, image_values = features["image"]
        mask_kind, mask_values = features["mask"]
        pose_kind, pose_values = features["camera_pose"]
        if (
            image_kind != "bytes"
            or mask_kind != "bytes"
            or pose_kind != "float"
            or len(image_values) != 1
            or len(mask_values) != 1
        ):
            raise ValueError(f"unexpected feature type at row {index}")
        metrics.append(
            sequence_signal(image_values[0], mask_values[0], pose_values, np)
        )
    if len(metrics) != entry["records"] or len(metrics) != len(expected_rows):
        raise ValueError("source record count differs from pinned audit")

    def summarize(field: str) -> dict[str, float]:
        return _percentiles([row[field] for row in metrics], np)

    pair_fields = (
        "rgb_adjacent_mae_mean",
        "mask_changed_pixel_fraction_mean",
        "camera_pose_delta_l2_mean",
        "mask_foreground_fraction",
    )
    return len(metrics), {
        "record_count": len(metrics),
        "videos_with_any_rgb_motion": int(
            sum(row["rgb_adjacent_mae_mean"] > 0 for row in metrics)
        ),
        "videos_with_any_mask_change": int(
            sum(row["mask_changed_transition_count"] > 0 for row in metrics)
        ),
        "videos_with_any_camera_pose_change": int(
            sum(row["camera_pose_delta_l2_mean"] > 0 for row in metrics)
        ),
        "all_15_rgb_frames_unique_count_distribution": summarize("unique_rgb_frames"),
        "all_15_mask_frames_unique_count_distribution": summarize("unique_mask_frames"),
        "all_15_camera_pose_rows_unique_count_distribution": summarize("unique_camera_pose_rows"),
        "per_video_temporal_signal_percentiles": {
            field: summarize(field) for field in pair_fields
        },
        "rgb_identical_transition_count_total": int(
            sum(row["rgb_identical_transition_count"] for row in metrics)
        ),
        "camera_pose_static_transition_count_total": int(
            sum(row["camera_pose_static_transition_count"] for row in metrics)
        ),
        "mask_value_cardinality_distribution": summarize("mask_unique_value_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_report = json.loads(args.train_audit.read_text(encoding="utf-8"))
    validation_report = json.loads(args.validation_audit.read_text(encoding="utf-8"))
    import numpy as np

    _, train = _load_split(manifest["train"], train_report, np)
    _, validation = _load_split(manifest["validation"], validation_report, np)
    result = {
        "status": "complete_label_free_temporal_data_screen",
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "train_audit_sha256": hashlib.sha256(args.train_audit.read_bytes()).hexdigest(),
        "validation_audit_sha256": hashlib.sha256(args.validation_audit.read_bytes()).hexdigest(),
        "metric_definitions": {
            "rgb_adjacent_mae_mean": (
                "mean absolute adjacent-frame RGB pixel difference divided by 255"
            ),
            "mask_changed_pixel_fraction_mean": (
                "mean fraction of pixels whose uint8 mask value changes between adjacent frames"
            ),
            "camera_pose_delta_l2_mean": (
                "mean L2 difference between adjacent six-value float32 camera-pose rows"
            ),
            "foreground": "fraction of mask pixels with value greater than zero",
        },
        "splits": {"train": train, "validation": validation},
        "probe_data_opened": False,
        "labels_used": False,
        "interpretation_limit": (
            "Temporal variation is measured, not evidence that a model can predict "
            "or classify physical events."
        ),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    print(f"receipt_sha256={hashlib.sha256(serialized.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
