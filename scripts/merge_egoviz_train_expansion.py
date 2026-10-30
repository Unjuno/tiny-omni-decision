"""Merge separately validated EgoViz training-only features with the fixed corpus."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from scripts.aggregate_egoviz_unseen_group_features import sha256_file

ORIGINAL_REPORT_SHA = "2655a6bdb654e7ab11b2c343286c2a2e007170412a5b554689843e6262381aa8"
EXPANSION_PLAN_SHA = "18958dcece44ae8a387895cbbe5b64ffeb218f4988a8d59401522d797f8703b3"


def verify_development_features(rows, old_video, old_sensor, merged_video, merged_sensor):
    old_index = {sample_id: i for i, sample_id in enumerate(old_video["sample_id"])}
    merged_index = {sample_id: i for i, sample_id in enumerate(merged_video["sample_id"])}
    for row in rows:
        sample_id = row["sample_id"]
        if (
            sample_id not in old_index
            or sample_id not in merged_index
            or not np.array_equal(
                old_video["features"][old_index[sample_id]],
                merged_video["features"][merged_index[sample_id]],
            )
        ):
            raise ValueError("development video features differ")
        for index in row["candidate_window_indices"]:
            if index < 0 or index >= len(old_sensor) or index >= len(merged_sensor):
                raise ValueError("invalid development sensor index")
            if not np.array_equal(old_sensor[index], merged_sensor[index]):
                raise ValueError("development sensor features differ")


def validate_separation(current, extra):
    if any(row["split"] != "train" for row in extra):
        raise ValueError("expansion must contain training only")
    if any(row["split"] not in ("train", "development") for row in current):
        raise ValueError("unexpected current split")
    train = [row for row in current if row["split"] == "train"]
    development = [row for row in current if row["split"] == "development"]
    for key in ("group_id", "chunk_id", "sample_id"):
        old_train = {row[key] for row in train}
        old_dev = {row[key] for row in development}
        new_train = {row[key] for row in extra}
        if old_train & old_dev or (old_train | old_dev) & new_train:
            raise ValueError(f"{key} overlaps across sources or splits")
    ids = [row["sample_id"] for row in [*current, *extra]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample_id")


def validate_references(rows, sessions):
    index = {}
    offset = 0
    for entry in sessions:
        if entry["chunk_id"] in index:
            raise ValueError("duplicate session chunk")
        index[entry["chunk_id"]] = (
            offset,
            entry["window_count"],
            entry["group_id"],
            entry["split"],
        )
        offset += entry["window_count"]
    for row in rows:
        start, count, group, split = index[row["chunk_id"]]
        candidates, target = row["candidate_window_indices"], row["target_option_index"]
        if (
            row["group_id"] != group
            or row["split"] != split
            or len(candidates) != 4
            or len(set(candidates)) != 4
            or not 0 <= target < 4
            or not all(start <= value < start + count for value in candidates)
            or candidates[target] != start + row["query_index"]
            or any(
                abs(value - (start + row["query_index"])) < 3
                for index, value in enumerate(candidates)
                if index != target
            )
        ):
            raise ValueError("invalid candidate/target session reference")


def _read_aggregate(root: Path):
    report_path = root / "aggregate-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report["status"] != "FROZEN_FEATURES_AGGREGATED_NO_TRAINING_OR_QUALITY_EVALUATION"
        or report["sealed_audit_accessed"] is not False
        or report["publisher_action_labels_read"] is not False
    ):
        raise ValueError("unverified or contaminated aggregate")
    for filename, expected in report["sha256"].items():
        if sha256_file(root / filename) != expected:
            raise ValueError(f"aggregate hash mismatch: {filename}")
    rows = [
        json.loads(line)
        for split in ("train", "development")
        for line in (root / f"{split}-samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validate_references(rows, report["sessions"])
    with np.load(root / "video-features.npz", allow_pickle=False) as archive:
        video = {key: archive[key].copy() for key in archive.files}
    with np.load(root / "sensor-features.npz", allow_pickle=False) as archive:
        sensor = {key: archive[key].copy() for key in archive.files}
    windows = np.load(root / "imu-windows.npy", allow_pickle=False)
    if (
        len(windows) != report["unique_imu_windows"]
        or not np.array_equal(sensor["window_index"], np.arange(len(windows)))
        or sensor["features"].shape != (len(windows), 6, 16, 512)
        or video["features"].shape != (len(rows), 8, 768)
        or set(video["sample_id"].tolist()) != {row["sample_id"] for row in rows}
        or len(set(video["sample_id"].tolist())) != len(rows)
        or not np.isfinite(video["features"]).all()
        or not np.isfinite(sensor["features"]).all()
    ):
        raise ValueError("invalid frozen aggregate arrays")
    return report, rows, video, sensor, windows


def merge(original: Path, expansion: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(output.parent).free < 3_000_000_000:
        raise RuntimeError("less than 3 GB free disk reserve")
    if sha256_file(original / "aggregate-report.json") != ORIGINAL_REPORT_SHA:
        raise ValueError("original frozen aggregate report mismatch")
    old, old_rows, old_video, old_sensor, old_windows = _read_aggregate(original)
    new, new_rows, new_video, new_sensor, new_windows = _read_aggregate(expansion)
    if (
        old["train_sessions"] != 24
        or old["development_sessions"] != 20
        or old["train_examples"] != 332
        or old["development_examples"] != 307
        or new["train_sessions"] != 24
        or new["development_sessions"] != 0
        or new["plan_sha256"] != EXPANSION_PLAN_SHA
    ):
        raise ValueError("unexpected source split counts or expansion plan")
    for key in (
        "video_checkpoint_sha256",
        "video_source_commit",
        "sensor_checkpoint_sha256",
        "sensor_source_commit",
    ):
        if old[key] != new[key]:
            raise ValueError(f"encoder revision mismatch: {key}")
    validate_separation(old_rows, new_rows)
    old_media = {entry["raw_media_sha256"] for entry in old["sessions"]}
    new_media = {entry["raw_media_sha256"] for entry in new["sessions"]}
    if len(old_media) != 44 or len(new_media) != 24 or old_media & new_media:
        raise ValueError("media SHA identities overlap or are duplicated")
    offset = len(old_windows)
    adjusted = [
        dict(row, candidate_window_indices=[i + offset for i in row["candidate_window_indices"]])
        for row in new_rows
    ]
    output.mkdir(parents=True)
    with (output / "train-samples.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in [*(row for row in old_rows if row["split"] == "train"), *adjusted]:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    shutil.copyfile(original / "development-samples.jsonl", output / "development-samples.jsonl")
    if (
        sha256_file(output / "development-samples.jsonl")
        != old["sha256"]["development-samples.jsonl"]
    ):
        raise ValueError("development ID/order bytes changed")
    np.save(
        output / "imu-windows.npy", np.concatenate([old_windows, new_windows]), allow_pickle=False
    )
    video = {key: np.concatenate([old_video[key], new_video[key]]) for key in old_video}
    order = np.argsort(video["sample_id"])
    np.savez_compressed(
        output / "video-features.npz", **{key: value[order] for key, value in video.items()}
    )
    features = np.concatenate([old_sensor["features"], new_sensor["features"]])
    np.savez_compressed(
        output / "sensor-features.npz", window_index=np.arange(len(features)), features=features
    )
    with np.load(output / "video-features.npz", allow_pickle=False) as saved_video:
        with np.load(output / "sensor-features.npz", allow_pickle=False) as saved_sensor:
            verify_development_features(
                [row for row in old_rows if row["split"] == "development"],
                old_video,
                old_sensor["features"],
                {key: saved_video[key] for key in ("sample_id", "features")},
                saved_sensor["features"],
            )
    report = {
        "status": "VALIDATED_TRAIN_EXPANSION_FIXED_DEVELOPMENT",
        "original_report_sha256": sha256_file(original / "aggregate-report.json"),
        "expansion_report_sha256": sha256_file(expansion / "aggregate-report.json"),
        "runner_sha256": sha256_file(Path(__file__)),
        "old_train_examples": 332,
        "new_train_examples": len(new_rows),
        "train_examples": 332 + len(new_rows),
        "development_examples": 307,
        "old_train_sessions": 24,
        "new_train_sessions": 24,
        "development_sessions": 20,
        "group_chunk_sample_media_overlap": 0,
        "source_media_count": 68,
        "candidate_target_references_validated": True,
        "development_bytes_unchanged": True,
        "development_video_and_sensor_features_unchanged": True,
        "sealed_audit_accessed": False,
        "category_counts_new": {
            category: sum(row["category"] == category for row in new_rows)
            for category in sorted({row["category"] for row in new_rows})
        },
        "sha256": {
            name: sha256_file(output / name)
            for name in (
                "train-samples.jsonl",
                "development-samples.jsonl",
                "imu-windows.npy",
                "video-features.npz",
                "sensor-features.npz",
            )
        },
    }
    (output / "merge-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.original, args.expansion, args.output), indent=2))


if __name__ == "__main__":
    main()
