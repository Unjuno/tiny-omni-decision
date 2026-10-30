"""Combine per-session EgoViz features into the fixed four-choice corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.download_egoviz_unseen_group_media import select_media_rows

VIDEO_CHECKPOINT_SHA256 = "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
VIDEO_SOURCE_COMMIT = "c6eba0ece3dd49a24ffc44582966997a4a59f1d8"
SENSOR_CHECKPOINT_SHA256 = "785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877"
SENSOR_SOURCE_COMMIT = "38f7310ad594100747ca2a8357e9c7ca7d323e0e"
FEATURE_STATUS = "FROZEN_FEATURES_EXTRACTED_NO_TRAINING_OR_QUALITY_EVALUATION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _verify_feature_report(
    report: dict[str, Any],
    cache_path: Path,
    *,
    modality: str,
    expected_count: int,
    expected_runner_sha256: str,
) -> None:
    if report.get("status") != FEATURE_STATUS or report.get("training") is not False:
        raise ValueError(f"{modality} feature report is not a frozen extraction")
    if report.get("sealed_audit_accessed") is not False:
        raise ValueError(f"{modality} feature report indicates sealed audit access")
    if report.get("feature_file_sha256") != sha256_file(cache_path):
        raise ValueError(f"{modality} feature cache hash does not match its report")
    if report.get("runner_sha256") != expected_runner_sha256:
        raise ValueError(f"{modality} feature extraction runner hash mismatch")
    count_key = "sample_count" if modality == "video" else "source_window_count"
    if report.get(count_key) != expected_count:
        raise ValueError(f"{modality} feature report count mismatch")


def _validate_chunk(
    plan_row: dict[str, Any],
    chunk_root: Path,
    *,
    plan_sha256: str,
    acquisition_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk_id = str(plan_row["chunk_id"])
    split = str(plan_row["split"])
    group_id = str(plan_row["group_id"])
    acquisition = acquisition_manifest or _read_json(chunk_root / "acquisition-manifest.json")
    files = acquisition.get("files")
    if (
        acquisition.get("format") != "egoviz-unseen-group-media-acquisition-v1"
        or acquisition.get("status") != "ACQUIRED_TRAIN_DEVELOPMENT_ONLY"
        or acquisition.get("plan_sha256") != plan_sha256
        or acquisition.get("sealed_audit_downloaded") is not False
        or not isinstance(files, list)
        or len(files) != 1
    ):
        raise ValueError(f"invalid acquisition receipt for {chunk_id}")
    receipt = files[0]
    for key, expected in (
        ("chunk_id", chunk_id),
        ("split", split),
        ("group_id", group_id),
        ("category", plan_row["category"]),
        ("mcap_path", plan_row["mcap_path"]),
        ("expected_bytes", int(plan_row["expected_bytes"])),
        ("bytes", int(plan_row["expected_bytes"])),
    ):
        if receipt.get(key) != expected:
            raise ValueError(f"acquisition receipt {key} mismatch for {chunk_id}")
    raw_sha256 = receipt.get("sha256")
    if not isinstance(raw_sha256, str) or len(raw_sha256) != 64:
        raise ValueError(f"missing downloaded media SHA-256 for {chunk_id}")

    corpus_dir = chunk_root / "window-corpus"
    corpus = _read_json(corpus_dir / "window-corpus.json")
    if (
        corpus.get("format") != "egoviz-vs-sync-window-corpus-v1"
        or corpus.get("seed") != 17
        or corpus.get("window_seconds") != 2
        or corpus.get("stride_seconds") != 1
        or corpus.get("video_frames_per_window") != 8
        or corpus.get("imu_points_per_window") != 128
        or corpus.get("sealed_audit_accessed") is not False
        or corpus.get("publisher_action_labels_read") is not False
        or corpus.get("group_overlap") != 0
    ):
        raise ValueError(f"invalid or contaminated window corpus for {chunk_id}")
    for filename, expected_hash in corpus.get("sha256", {}).items():
        path = corpus_dir / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"window corpus hash mismatch for {chunk_id}: {filename}")
    windows = np.load(corpus_dir / "imu-windows.npy", allow_pickle=False)
    if windows.ndim != 3 or windows.shape[1:] != (6, 128) or not np.isfinite(windows).all():
        raise ValueError(f"invalid IMU windows for {chunk_id}: {windows.shape}")
    window_count = len(windows)
    inventory = corpus.get("inventory")
    if (
        not isinstance(inventory, list)
        or len(inventory) != 1
        or inventory[0].get("chunk_id") != chunk_id
        or inventory[0].get("group_id") != group_id
        or inventory[0].get("split") != split
        or inventory[0].get("windows") != window_count
    ):
        raise ValueError(f"window-corpus inventory mismatch for {chunk_id}")

    rows_by_split = {
        name: _read_jsonl(corpus_dir / f"{name}-samples.jsonl")
        for name in ("train", "development")
    }
    expected_counts = {
        "train": int(corpus.get("train_examples", -1)),
        "development": int(corpus.get("development_examples", -1)),
    }
    if expected_counts[split] != len(rows_by_split[split]) or expected_counts[
        "development" if split == "train" else "train"
    ] != 0:
        raise ValueError(f"corpus sample count/split mismatch for {chunk_id}")
    samples = rows_by_split[split]
    if not samples:
        raise ValueError(f"no four-choice samples produced for {chunk_id}")
    for row in samples:
        if (
            row.get("split") != split
            or row.get("chunk_id") != chunk_id
            or row.get("group_id") != group_id
            or row.get("category") != plan_row["category"]
        ):
            raise ValueError(f"sample provenance mismatch for {chunk_id}")
        expected_sample_id = hashlib.sha256(
            f"{corpus['seed']}:{chunk_id}:{row['query_index']}".encode()
        ).hexdigest()[:24]
        if row.get("sample_id") != expected_sample_id:
            raise ValueError(f"non-deterministic sample ID for {chunk_id}")
        candidates = row.get("candidate_window_indices")
        target = row.get("target_option_index")
        valid_candidates = (
            isinstance(candidates, list)
            and len(candidates) == 4
            and all(type(index) is int and 0 <= index < window_count for index in candidates)
        )
        if (
            not valid_candidates
            or len(set(candidates)) != 4
            or type(target) is not int
            or not 0 <= target < 4
            or type(row.get("query_index")) is not int
            or candidates[target] != row["query_index"]
        ):
            raise ValueError(f"invalid four-choice target/window references for {chunk_id}")

    video_dir = chunk_root / "video-features"
    video_path = video_dir / "video-features.npz"
    video_report = _read_json(video_dir / "feature-report.json")
    if (
        video_report.get("source_checkpoint_sha256") != VIDEO_CHECKPOINT_SHA256
        or video_report.get("upstream_commit") != VIDEO_SOURCE_COMMIT
        or video_report.get("jax_version") != "0.6.2"
        or video_report.get("devices") != ["TFRT_CPU_0"]
    ):
        raise ValueError(f"unexpected VideoPrism checkpoint/source/runtime for {chunk_id}")
    video_runner_sha256 = sha256_file(
        Path(__file__).with_name("extract_egoviz_vs_sync_video_features.py")
    )
    _verify_feature_report(
        video_report,
        video_path,
        modality="video",
        expected_count=len(samples),
        expected_runner_sha256=video_runner_sha256,
    )
    video = _load_npz(video_path)
    ordered_samples = sorted(samples, key=lambda row: row["sample_id"])
    if (
        video.get("sample_id", np.empty(0)).tolist()
        != [row["sample_id"] for row in ordered_samples]
        or video.get("split", np.empty(0)).tolist() != [split] * len(samples)
        or video.get("group_id", np.empty(0)).tolist() != [group_id] * len(samples)
        or video.get("features", np.empty((0,))).shape != (len(samples), 8, 768)
        or not np.isfinite(video["features"]).all()
    ):
        raise ValueError(f"VideoPrism feature/sample alignment mismatch for {chunk_id}")

    sensor_dir = chunk_root / "sensor-features"
    sensor_path = sensor_dir / "sensor-features.npz"
    sensor_report = _read_json(sensor_dir / "feature-report.json")
    if (
        sensor_report.get("checkpoint_sha256") != SENSOR_CHECKPOINT_SHA256
        or sensor_report.get("moment_source_commit") != SENSOR_SOURCE_COMMIT
        or sensor_report.get("device") != "cpu"
        or sensor_report.get("torch_version") != "2.6.0+cu124"
        or sensor_report.get("transformers_version") != "5.6.2"
    ):
        raise ValueError(f"unexpected MOMENT checkpoint/source/runtime for {chunk_id}")
    sensor_runner_sha256 = sha256_file(
        Path(__file__).with_name("extract_egoviz_vs_sync_sensor_features.py")
    )
    _verify_feature_report(
        sensor_report,
        sensor_path,
        modality="sensor",
        expected_count=window_count,
        expected_runner_sha256=sensor_runner_sha256,
    )
    sensor = _load_npz(sensor_path)
    if (
        sensor.get("window_index", np.empty(0)).tolist() != list(range(window_count))
        or sensor.get("features", np.empty((0,))).shape != (window_count, 6, 16, 512)
        or not np.isfinite(sensor["features"]).all()
    ):
        raise ValueError(f"MOMENT feature/window alignment mismatch for {chunk_id}")

    return {
        "rows": samples,
        "windows": windows,
        "video_features": video["features"],
        "sensor_features": sensor["features"],
        "raw_sha256": raw_sha256,
        "video_cache_sha256": sha256_file(video_path),
        "sensor_cache_sha256": sha256_file(sensor_path),
        "inventory": inventory[0],
    }


def aggregate(plan_path: Path, runs_root: Path, output: Path) -> dict[str, Any]:
    """Validate every selected session and combine its immutable cached features."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    plan = _read_json(plan_path)
    plan_hash = sha256_file(plan_path)
    expected = select_media_rows(plan, splits=("train", "development"))
    expected_ids = {str(row["chunk_id"]) for row in expected}
    actual_ids = {path.name for path in runs_root.iterdir() if path.is_dir()}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(f"missing planned sessions {missing}; unexpected session dirs {extra}")

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "development": []}
    window_parts: list[np.ndarray] = []
    sensor_parts: list[np.ndarray] = []
    video_by_id: dict[str, tuple[str, str, np.ndarray]] = {}
    session_records: list[dict[str, Any]] = []
    group_sets: dict[str, set[str]] = {"train": set(), "development": set()}
    window_offset = 0

    for plan_row in expected:
        chunk_id = str(plan_row["chunk_id"])
        validated = _validate_chunk(plan_row, runs_root / chunk_id, plan_sha256=plan_hash)
        split = str(plan_row["split"])
        group_id = str(plan_row["group_id"])
        if group_id in group_sets[split]:
            raise ValueError(f"duplicate {split} group in plan: {group_id}")
        group_sets[split].add(group_id)
        local_rows = validated["rows"]
        for row in local_rows:
            rebased = dict(row)
            rebased["candidate_window_indices"] = [
                index + window_offset for index in row["candidate_window_indices"]
            ]
            rows_by_split[split].append(rebased)
        ordered_rows = sorted(local_rows, key=lambda row: row["sample_id"])
        for row, features in zip(ordered_rows, validated["video_features"], strict=True):
            sample_id = str(row["sample_id"])
            if sample_id in video_by_id:
                raise ValueError(f"duplicate sample ID: {sample_id}")
            video_by_id[sample_id] = (split, group_id, features)
        window_parts.append(validated["windows"])
        sensor_parts.append(validated["sensor_features"])
        session_records.append(
            {
                "chunk_id": chunk_id,
                "group_id": group_id,
                "split": split,
                "category": str(plan_row["category"]),
                "sample_count": len(local_rows),
                "window_count": len(validated["windows"]),
                "video_frames": int(validated["inventory"]["video_frames"]),
                "imu_samples": int(validated["inventory"]["imu_samples"]),
                "raw_media_sha256": validated["raw_sha256"],
                "video_cache_sha256": validated["video_cache_sha256"],
                "sensor_cache_sha256": validated["sensor_cache_sha256"],
            }
        )
        window_offset += len(validated["windows"])

    group_overlap = group_sets["train"] & group_sets["development"]
    if group_overlap:
        raise ValueError(f"train/development group overlap: {sorted(group_overlap)}")
    for split in rows_by_split:
        rows_by_split[split].sort(key=lambda row: row["sample_id"])
    if len(video_by_id) != sum(len(rows) for rows in rows_by_split.values()):
        raise ValueError("aggregate video feature count does not match sample count")

    windows_all = np.concatenate(window_parts, axis=0)
    sensor_all = np.concatenate(sensor_parts, axis=0)
    ordered_ids = sorted(video_by_id)
    video_all = np.stack([video_by_id[sample_id][2] for sample_id in ordered_ids])
    video_splits = [video_by_id[sample_id][0] for sample_id in ordered_ids]
    video_groups = [video_by_id[sample_id][1] for sample_id in ordered_ids]
    if not np.isfinite(windows_all).all() or not np.isfinite(sensor_all).all() or not np.isfinite(
        video_all
    ).all():
        raise ValueError("non-finite value in aggregate feature arrays")

    output.mkdir(parents=True, exist_ok=False)
    for split, rows in rows_by_split.items():
        with (output / f"{split}-samples.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (output / "imu-windows.npy").open("xb") as stream:
        np.save(stream, windows_all, allow_pickle=False)
    np.savez_compressed(
        output / "video-features.npz",
        sample_id=np.asarray(ordered_ids),
        split=np.asarray(video_splits),
        group_id=np.asarray(video_groups),
        features=video_all,
    )
    np.savez_compressed(
        output / "sensor-features.npz",
        window_index=np.arange(len(sensor_all), dtype=np.int32),
        features=sensor_all,
    )
    source_meta = plan.get("source", {})
    split_seed = plan.get("seed")
    if split_seed is None and isinstance(source_meta, dict):
        split_seed = source_meta.get("seed")
    payload: dict[str, Any] = {
        "format": "egoviz-unseen-session-aggregate-v1",
        "status": "FROZEN_FEATURES_AGGREGATED_NO_TRAINING_OR_QUALITY_EVALUATION",
        "plan_sha256": plan_hash,
        "split_seed": int(split_seed) if split_seed is not None else None,
        "window_sampling_seed": 17,
        "session_count": len(session_records),
        "train_sessions": len(group_sets["train"]),
        "development_sessions": len(group_sets["development"]),
        "train_examples": len(rows_by_split["train"]),
        "development_examples": len(rows_by_split["development"]),
        "unique_imu_windows": len(windows_all),
        "video_feature_shape": list(video_all.shape),
        "sensor_feature_shape": list(sensor_all.shape),
        "train_development_group_overlap": len(group_overlap),
        "video_checkpoint_sha256": VIDEO_CHECKPOINT_SHA256,
        "video_source_commit": VIDEO_SOURCE_COMMIT,
        "sensor_checkpoint_sha256": SENSOR_CHECKPOINT_SHA256,
        "sensor_source_commit": SENSOR_SOURCE_COMMIT,
        "publisher_action_labels_read": False,
        "sealed_audit_accessed": False,
        "training": False,
        "sessions": session_records,
        "sha256": {
            name: sha256_file(output / name)
            for name in (
                "imu-windows.npy",
                "train-samples.jsonl",
                "development-samples.jsonl",
                "video-features.npz",
                "sensor-features.npz",
            )
        },
    }
    corpus_payload = {
        "format": "egoviz-vs-sync-window-corpus-v1",
        "seed": 17,
        "window_seconds": 2,
        "stride_seconds": 1,
        "video_frames_per_window": 8,
        "imu_points_per_window": 128,
        "imu_channels": [
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        ],
        "train_examples": len(rows_by_split["train"]),
        "development_examples": len(rows_by_split["development"]),
        "train_groups": len(group_sets["train"]),
        "development_groups": len(group_sets["development"]),
        "group_overlap": 0,
        "unique_imu_windows": len(windows_all),
        "tensor_shape": list(windows_all.shape),
        "publisher_action_labels_read": False,
        "sealed_audit_accessed": False,
        "inventory": session_records,
        "sha256": {
            name: sha256_file(output / name)
            for name in (
                "imu-windows.npy",
                "train-samples.jsonl",
                "development-samples.jsonl",
            )
        },
    }
    with (output / "window-corpus.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(corpus_payload, stream, indent=2)
        stream.write("\n")
    with (output / "aggregate-report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return payload


def validate_session(
    plan_path: Path,
    runs_root: Path,
    chunk_id: str,
    acquisition_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one planned train/development package without opening other sessions."""
    plan = _read_json(plan_path)
    plan_hash = sha256_file(plan_path)
    selected = [
        row
        for row in select_media_rows(plan, splits=("train", "development"))
        if str(row["chunk_id"]) == chunk_id
    ]
    if len(selected) != 1:
        raise ValueError(f"chunk is not uniquely present in the train/development plan: {chunk_id}")
    receipt_override = None
    if acquisition_manifest_path is not None:
        shared_receipt = _read_json(acquisition_manifest_path)
        expected_rows = {
            str(row["chunk_id"]): row
            for row in select_media_rows(plan, splits=("train", "development"))
        }
        receipt_files = shared_receipt.get("files")
        if (
            shared_receipt.get("format") != "egoviz-unseen-group-media-acquisition-v1"
            or shared_receipt.get("status") != "ACQUIRED_TRAIN_DEVELOPMENT_ONLY"
            or shared_receipt.get("plan_sha256") != plan_hash
            or shared_receipt.get("sealed_audit_downloaded") is not False
            or not isinstance(receipt_files, list)
        ):
            raise ValueError("invalid shared acquisition receipt for frozen plan")
        seen: set[str] = set()
        selected_receipts: list[dict[str, Any]] = []
        for receipt in receipt_files:
            receipt_id = str(receipt.get("chunk_id", ""))
            planned = expected_rows.get(receipt_id)
            if planned is None or receipt_id in seen:
                raise ValueError(
                    f"shared receipt contains unknown or duplicate chunk: {receipt_id}"
                )
            seen.add(receipt_id)
            for key, expected in (
                ("split", planned["split"]),
                ("group_id", planned["group_id"]),
                ("category", planned["category"]),
                ("mcap_path", planned["mcap_path"]),
                ("expected_bytes", int(planned["expected_bytes"])),
                ("bytes", int(planned["expected_bytes"])),
            ):
                if receipt.get(key) != expected:
                    raise ValueError(f"shared receipt {key} mismatch for {receipt_id}")
            digest = receipt.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"shared receipt lacks media SHA-256 for {receipt_id}")
            if receipt_id == chunk_id:
                selected_receipts.append(receipt)
        if len(selected_receipts) != 1:
            raise ValueError(f"shared receipt does not uniquely contain planned chunk: {chunk_id}")
        receipt_override = {
            "format": shared_receipt["format"],
            "status": shared_receipt["status"],
            "plan_sha256": shared_receipt["plan_sha256"],
            "sealed_audit_downloaded": False,
            "files": selected_receipts,
        }
    validated = _validate_chunk(
        selected[0],
        runs_root / chunk_id,
        plan_sha256=plan_hash,
        acquisition_manifest=receipt_override,
    )
    row = selected[0]
    return {
        "status": "FROZEN_SESSION_PACKAGE_VALIDATED_NO_TRAINING_OR_QUALITY_EVALUATION",
        "plan_sha256": plan_hash,
        "chunk_id": chunk_id,
        "group_id": str(row["group_id"]),
        "split": str(row["split"]),
        "sample_count": len(validated["rows"]),
        "window_count": len(validated["windows"]),
        "raw_media_sha256": validated["raw_sha256"],
        "video_feature_sha256": validated["video_cache_sha256"],
        "sensor_feature_sha256": validated["sensor_cache_sha256"],
        "publisher_action_labels_read": False,
        "sealed_audit_accessed": False,
        "training": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate-chunk-id",
        help="validate only this planned train/development session package",
    )
    parser.add_argument(
        "--acquisition-manifest",
        type=Path,
        help="optional shared full-plan receipt when package receipts were split",
    )
    args = parser.parse_args()
    if args.validate_chunk_id:
        if args.output is not None:
            parser.error("--output cannot be combined with --validate-chunk-id")
        result = validate_session(
            args.plan,
            args.runs_root,
            args.validate_chunk_id,
            args.acquisition_manifest,
        )
    else:
        if args.output is None or args.acquisition_manifest is not None:
            parser.error("--output is required unless --validate-chunk-id is used")
        result = aggregate(args.plan, args.runs_root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
