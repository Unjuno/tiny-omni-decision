"""Analyze a receipt of HapTile train-only MCAP samples; never reads held-out splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.haptile_action import (  # noqa: E402
    ACTION_LABELS,
    DEFAULT_MOTION_DEADBAND_M,
    directional_action_label,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode())  # noqa: S324 - source OID check
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_train_receipt(
    receipt: dict[str, object], split_manifest: dict[str, object], split_hash: str
) -> list[dict[str, object]]:
    if receipt.get("format") not in {
        "haptile_train_only_sample_preflight_v1",
        "haptile_frozen_train_corpus_v1",
    }:
        raise ValueError("input is not the expected train-only HapTile receipt")
    revision = "c8593da98d3facb7a9293842514f0ee4ad3f2318"
    if receipt.get("revision", receipt.get("conversion_revision")) != revision:
        raise ValueError("HapTile conversion revision differs from the pinned release")
    if receipt.get(
        "split_manifest_sha256", receipt.get("split_manifest_file_sha256")
    ) != split_hash:
        raise ValueError("receipt uses a different task split manifest")
    splits = split_manifest.get("development_splits")
    if not isinstance(splits, dict) or not isinstance(splits.get("train"), list):
        raise ValueError("public split manifest lacks train task groups")
    train_tasks = set(splits["train"])
    heldout_tasks = set(splits.get("calibration", [])) | set(splits.get("validation", []))
    if train_tasks & heldout_tasks:
        raise ValueError("public split manifest overlaps train and held-out groups")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("train receipt must contain at least one file")
    if receipt.get("format") == "haptile_frozen_train_corpus_v1":
        if len(files) != 1113 or receipt.get("episode_count") != 1113:
            raise ValueError("full train-corpus receipt must contain all 1,113 episodes")
        task_ids = {
            Path(str(item.get("path", ""))).parts[1]
            for item in files
            if isinstance(item, dict) and len(Path(str(item.get("path", ""))).parts) == 3
        }
        if task_ids != train_tasks:
            raise ValueError("full train-corpus receipt does not cover exactly the train tasks")
        paths = [item.get("path") for item in files if isinstance(item, dict)]
        if len(paths) != len(set(paths)):
            raise ValueError("full train-corpus receipt contains duplicate paths")
    validated: list[dict[str, object]] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("receipt contains a malformed file row")
        rel = Path(str(item.get("path", "")))
        task = item.get("task")
        if task is None and receipt.get("format") == "haptile_frozen_train_corpus_v1":
            task = rel.parts[1] if len(rel.parts) == 3 else None
        if task not in train_tasks:
            raise ValueError("receipt contains a non-train task")
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or rel.suffix != ".mcap"
            or len(rel.parts) != 3
            or rel.parts[0] != "data"
            or rel.parts[1] != task
        ):
            raise ValueError("receipt contains an unsafe path or task/path mismatch")
        validated.append(item)
    return validated


def analyze_receipt(
    receipt_path: Path,
    output_path: Path,
    split_manifest_path: Path,
    *,
    audit_gripper_windows: bool = False,
) -> dict[str, object]:
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:
        raise RuntimeError("install pinned mcap dependencies into an isolated environment") from exc

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_hash = _sha256(split_manifest_path)
    files = validate_train_receipt(receipt, split_manifest, split_hash)
    revision = str(receipt.get("revision", receipt.get("conversion_revision")))
    counts: Counter[str] = Counter()
    gripper_midpoint_counts: Counter[str] = Counter()
    gripper_window_counts: dict[str, Counter[str]] = {
        str(threshold): Counter() for threshold in (0.1, 0.05, 0.02, 0.01)
    }
    rows: list[dict[str, object]] = []
    total_messages = 0
    total_gripper_windows = 0

    for item in files:
        rel = Path(str(item["path"]))
        task = item.get("task", rel.parts[1])
        path = receipt_path.parent / rel
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise ValueError(f"missing or size-mismatched train sample: {rel}")
        digest = _sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"train sample SHA-256 mismatch: {rel}")
        if item.get("lfs_sha256") is not None and digest != item["lfs_sha256"]:
            raise ValueError(f"train sample LFS SHA-256 mismatch: {rel}")
        if (
            item.get("lfs_sha256") is None
            and _git_blob_sha1(path) != item.get("git_blob_oid")
        ):
            raise ValueError(f"train sample Git blob OID mismatch: {rel}")

        times: list[float] = []
        positions: list[tuple[float, float, float]] = []
        with path.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()], validate_crcs=True)
            for schema, channel, message, decoded in reader.iter_decoded_messages(
                topics=["/end-effector-pose"]
            ):
                if schema is None or schema.name != "foxglove.PoseInFrame":
                    raise ValueError(f"unexpected pose schema for {rel}")
                if channel.topic != "/end-effector-pose":
                    raise ValueError(f"unexpected pose topic for {rel}")
                timestamp = message.log_time / 1_000_000_000
                position = decoded.pose.position
                times.append(timestamp)
                positions.append((position.x, position.y, position.z))
        if len(times) < 4 or any(
            b <= a for a, b in zip(times, times[1:], strict=False)
        ):
            raise ValueError(f"insufficient or non-monotonic pose stream: {rel}")
        query = (times[0] + times[-1]) / 2
        current = [
            position for timestamp, position in zip(times, positions, strict=True)
            if query - 0.25 <= timestamp <= query
        ]
        future = [
            position for timestamp, position in zip(times, positions, strict=True)
            if query + 0.25 <= timestamp <= query + 0.50
        ]
        if not current or not future:
            raise ValueError(f"midpoint does not have the fixed target window: {rel}")
        current_xyz = tuple(statistics.median(row[axis] for row in current) for axis in range(3))
        future_xyz = tuple(statistics.median(row[axis] for row in future) for axis in range(3))
        label = directional_action_label(current_xyz, future_xyz)
        counts[label] += 1

        gripper_delta: float | None = None
        if audit_gripper_windows:
            gripper_times: list[float] = []
            gripper_values: list[float] = []
            with path.open("rb") as stream:
                reader = make_reader(stream, validate_crcs=True)
                for schema, channel, message in reader.iter_messages(
                    topics=["/command.plot"]
                ):
                    if (
                        channel.topic != "/command.plot"
                        or schema is None
                        or schema.encoding != "jsonschema"
                    ):
                        raise ValueError(f"unexpected command schema for {rel}")
                    command = json.loads(message.data)
                    if "gripper" not in command:
                        raise ValueError(f"command stream has no gripper value: {rel}")
                    value = float(command["gripper"])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"command stream has a non-finite gripper value: {rel}"
                        )
                    gripper_times.append(message.log_time / 1_000_000_000)
                    gripper_values.append(value)
            gripper_before = [
                value for timestamp, value in zip(
                    gripper_times, gripper_values, strict=True
                )
                if query - 0.5 <= timestamp <= query - 0.1
            ]
            gripper_after = [
                value for timestamp, value in zip(
                    gripper_times, gripper_values, strict=True
                )
                if query + 0.1 <= timestamp <= query + 0.5
            ]
            if not gripper_before or not gripper_after:
                raise ValueError(f"midpoint does not have the fixed gripper window: {rel}")
            gripper_delta = statistics.median(gripper_after) - statistics.median(gripper_before)
            gripper_midpoint_counts[
                "increase" if gripper_delta >= 0.1 else
                "decrease" if gripper_delta <= -0.1 else "hold"
            ] += 1
            for timestamp in gripper_times:
                before = [
                    value for sample_time, value in zip(
                        gripper_times, gripper_values, strict=True
                    )
                    if timestamp - 0.5 <= sample_time <= timestamp
                ]
                after = [
                    value for sample_time, value in zip(
                        gripper_times, gripper_values, strict=True
                    )
                    if timestamp + 0.1 <= sample_time <= timestamp + 0.5
                ]
                if not before or not after:
                    continue
                delta = statistics.median(after) - statistics.median(before)
                total_gripper_windows += 1
                for threshold in gripper_window_counts:
                    numeric_threshold = float(threshold)
                    category = (
                        "increase" if delta >= numeric_threshold else
                        "decrease" if delta <= -numeric_threshold else "hold"
                    )
                    gripper_window_counts[threshold][category] += 1
        total_messages += len(times)
        row: dict[str, object] = {
            "task": task,
            "path": item["path"],
            "sha256": digest,
            "pose_messages": len(times),
            "episode_duration_seconds": round(times[-1] - times[0], 6),
            "query_time_log_seconds": round(query, 6),
            "current_xyz_m": current_xyz,
            "future_xyz_m": future_xyz,
            "target": label,
        }
        if audit_gripper_windows:
            row["provisional_gripper_command_delta_midpoint"] = gripper_delta
            row["provisional_gripper_target_at_0.10"] = (
                "increase" if gripper_delta >= 0.1 else
                "decrease" if gripper_delta <= -0.1 else "hold"
            )
        rows.append(row)

    output = {
        "format": "haptile_train_only_action_preflight_v1",
        "conversion_revision": revision,
        "input_receipt_sha256": _sha256(receipt_path),
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": receipt.get(
            "split_manifest_sha256", receipt.get("split_manifest_file_sha256")
        ),
        "split_accessed": "train only",
        "query_protocol": {
            "one_query_per_episode": "episode midpoint",
            "input_cutoff": "midpoint, inclusive",
            "target_current_pose_window": "[t-0.25s, t]",
            "target_future_pose_window": "[t+0.25s, t+0.50s]",
            "target": "dominant Cartesian displacement direction, else hold",
            "deadband_m": DEFAULT_MOTION_DEADBAND_M,
            "labels": list(ACTION_LABELS),
        },
        "sample_count": len(rows),
        "pose_messages": total_messages,
        "class_counts": {label: counts[label] for label in ACTION_LABELS},
        "rejected_gripper_target_preflight": ({
            "channel": "/command.plot.gripper",
            "midpoint_class_counts_at_0.10": {
                label: gripper_midpoint_counts[label]
                for label in ("increase", "decrease", "hold")
            },
            "all_overlapping_windows": total_gripper_windows if audit_gripper_windows else None,
            "all_window_class_counts_by_absolute_delta_threshold": ({
                threshold: {
                    label: gripper_window_counts[threshold][label]
                    for label in ("increase", "decrease", "hold")
                }
                for threshold in gripper_window_counts
            } if audit_gripper_windows else None),
            "window_protocol": "current command median [t-0.5s,t], future median [t+0.1s,t+0.5s]",
        } if audit_gripper_windows else None),
        "samples": rows,
        "validation_or_sealed_audit_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-manifest", type=Path,
        default=ROOT / "manifests/pretrained_reuse/haptile-task-split-v1-20261122.json",
    )
    parser.add_argument("--audit-gripper-windows", action="store_true")
    args = parser.parse_args()
    result = analyze_receipt(
        args.receipt, args.output, args.split_manifest,
        audit_gripper_windows=args.audit_gripper_windows,
    )
    print(json.dumps({
        "samples": result["sample_count"],
        "class_counts": result["class_counts"],
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
