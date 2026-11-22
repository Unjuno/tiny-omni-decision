"""Join frozen HapTile video, causal state, text and target rows by media ID."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tiny_omni_decision.haptile_action import ACTION_LABELS  # noqa: E402

SPLITS = ("train", "validation", "calibration")
VIDEO_STREAMS = ("/front-camera", "/wrist-camera", "/tactile-left", "/tactile-right")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def align_split_features(
    *,
    action_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    text_cache: dict[str, np.ndarray],
    video_values: np.ndarray,
    state_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Prove one-to-one path/hash/target alignment before training or eval."""
    count = len(action_rows)
    if count < 1 or len(state_rows) != count or len(video_rows) != count:
        raise ValueError("action, state and video row counts differ")
    if video_values.shape != (count, 4, 8, 768):
        raise ValueError("VideoPrism feature tensor has an unexpected shape")
    if state_values.shape != (count, 8, 25):
        raise ValueError("causal state tensor has an unexpected shape")
    paths = [str(row.get("path", "")) for row in action_rows]
    if not all(paths) or len(paths) != len(set(paths)):
        raise ValueError("action rows require unique stable media IDs")
    context = np.asarray(text_cache.get("context_features"), dtype=np.float32)
    options = np.asarray(text_cache.get("option_features"), dtype=np.float32)
    context_no_instruction = np.asarray(
        text_cache.get("context_features_no_instruction"), dtype=np.float32
    )
    options_no_instruction = np.asarray(
        text_cache.get("option_features_no_instruction"), dtype=np.float32
    )
    text_paths = [str(value) for value in text_cache.get("sample_paths", [])]
    text_hashes = [str(value) for value in text_cache.get("media_sha256", [])]
    text_targets = [str(value) for value in text_cache.get("targets", [])]
    if (context.shape != (count, 384) or options.shape != (count, 7, 1536)
            or context_no_instruction.shape != (count, 384)
            or options_no_instruction.shape != (count, 7, 1536)
            or len(text_paths) != count or len(text_hashes) != count
            or len(text_targets) != count
            or not np.isfinite(context).all() or not np.isfinite(options).all()
            or not np.isfinite(context_no_instruction).all()
            or not np.isfinite(options_no_instruction).all()
            or not np.isfinite(video_values).all() or not np.isfinite(state_values).all()):
        raise ValueError("feature cache shapes, IDs or finite values are invalid")

    state_by_path = {str(row.get("path")): row for row in state_rows}
    video_by_path = {str(row.get("path")): row for row in video_rows}
    state_indices = {str(row.get("path")): i for i, row in enumerate(state_rows)}
    video_indices = {str(row.get("path")): i for i, row in enumerate(video_rows)}
    text_indices = {path: i for i, path in enumerate(text_paths)}
    if (len(state_by_path) != count or len(video_by_path) != count
            or len(text_indices) != count):
        raise ValueError("state or video features contain duplicate media IDs")
    if (set(state_by_path) != set(paths) or set(video_by_path) != set(paths)
            or set(text_indices) != set(paths)):
        raise ValueError("feature caches contain missing or extra media IDs")
    if (len(text_paths) != len(set(text_paths))
            or len(text_hashes) != count or len(text_targets) != count):
        raise ValueError("text feature cache has duplicate or incomplete rows")

    task_groups: list[str] = []
    targets: list[int] = []
    state_order: list[int] = []
    video_order: list[int] = []
    text_order: list[int] = []
    video_times = np.empty((count, 8), dtype=np.float32)
    state_times = np.tile(np.linspace(-1000.0, 0.0, 8, dtype=np.float32), (count, 1))
    for index, target_row in enumerate(action_rows):
        path = paths[index]
        state_row, video_row = state_by_path[path], video_by_path[path]
        text_index = text_indices[path]
        digest, target = str(target_row.get("sha256")), str(target_row.get("target"))
        if (state_row.get("sha256") != digest or video_row.get("sha256") != digest
                or text_hashes[text_index] != digest or text_targets[text_index] != target
                or state_row.get("target") != target or video_row.get("target") != target):
            raise ValueError(f"media hash or target differs across feature sources: {path}")
        if target not in ACTION_LABELS:
            raise ValueError(f"unknown action target: {target}")
        group = str(target_row.get("task", state_row.get("task_group", "")))
        if not group or state_row.get("task_group", group) != group:
            raise ValueError(f"task-group identity differs across feature sources: {path}")
        selected = video_row.get("selected_frame_times_log_seconds")
        query = float(target_row["query_time_log_seconds"])
        if not isinstance(selected, dict) or set(selected) != set(VIDEO_STREAMS):
            raise ValueError(f"video timestamps do not cover all frozen streams: {path}")
        stream_times: list[list[float]] = []
        for stream in VIDEO_STREAMS:
            times = [float(value) for value in selected[stream]]
            if (len(times) != 8 or any(not np.isfinite(value) for value in times)
                    or any(value > query for value in times)):
                raise ValueError(f"video timestamps are invalid or noncausal: {path}")
            stream_times.append(times)
        # Each frame index is sampled before the corresponding shared target
        # time. Averaging stream log-times yields a causal shared timestamp.
        video_times[index] = (
            np.mean(np.asarray(stream_times, dtype=np.float64), axis=0) - query
        ) * 1000.0
        if np.any(video_times[index, 1:] <= video_times[index, :-1]):
            raise ValueError(f"shared video timestamps are not strictly increasing: {path}")
        task_groups.append(group)
        targets.append(ACTION_LABELS.index(target))
        state_order.append(state_indices[path])
        video_order.append(video_indices[path])
        text_order.append(text_index)

    return {
        "sample_paths": np.asarray(paths, dtype=np.str_),
        "media_sha256": np.asarray([str(row["sha256"]) for row in action_rows], dtype=np.str_),
        "task_groups": np.asarray(task_groups, dtype=np.str_),
        "targets": np.asarray(targets, dtype=np.int64),
        "video_values": np.asarray(video_values[video_order], dtype=np.float32),
        "video_timestamps_ms": video_times,
        "state_values": np.asarray(state_values[state_order], dtype=np.float32),
        "state_timestamps_ms": state_times,
        "context_features": context[text_order],
        "option_features": options[text_order],
        "context_features_no_instruction": context_no_instruction[text_order],
        "option_features_no_instruction": options_no_instruction[text_order],
    }


def apply_video_window_eligibility(
    *, action_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]], text_cache: dict[str, np.ndarray],
    video_values: np.ndarray, state_values: np.ndarray, eligibility: dict[str, Any],
    split_name: str, receipt_sha256: str,
    action_sha256: str, split_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
           dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    """Select the same label-blind video-eligible IDs across every feature path."""
    expected = {
        "format": f"haptile_{split_name}_video_window_eligibility_v2",
        "split": split_name,
        "input_receipt_sha256": receipt_sha256,
        "action_preflight_sha256": action_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "selection_uses_labels": False,
        "sealed_audit_opened": False,
    }
    if any(eligibility.get(key) != value for key, value in expected.items()):
        raise ValueError("video-window eligibility provenance differs from joined split")
    if eligibility.get("criteria") != {
        "context_seconds": 1.0,
        "frames_per_stream": 8,
        "max_frame_time_error_seconds": 0.1,
        "causal_cutoff_inclusive": True,
        "streams": list(VIDEO_STREAMS),
        "excluded_only_for": "insufficient distinct causal frames in fixed window",
    }:
        raise ValueError("video-window eligibility policy differs from the frozen protocol")

    all_paths = [str(row.get("path", "")) for row in action_rows]
    if (not all_paths or not all(path for path in all_paths)
            or len(all_paths) != len(set(all_paths))):
        raise ValueError("action rows require unique media paths before eligibility filtering")
    action_by_path = {str(row["path"]): row for row in action_rows}
    eligible_rows = eligibility.get("eligible_samples")
    excluded_rows = eligibility.get("excluded_samples")
    if not isinstance(eligible_rows, list) or not isinstance(excluded_rows, list):
        raise ValueError("video-window eligibility rows are missing")
    eligible_by_path = {
        str(row.get("path")): row for row in eligible_rows if isinstance(row, dict)
    }
    excluded_by_path = {
        str(row.get("path")): row for row in excluded_rows if isinstance(row, dict)
    }
    eligible_paths = set(eligible_by_path)
    excluded_paths = set(excluded_by_path)
    if (len(eligible_by_path) != len(eligible_rows)
            or len(excluded_by_path) != len(excluded_rows)
            or eligible_paths & excluded_paths
            or eligible_paths | excluded_paths != set(all_paths)):
        raise ValueError("video-window eligibility does not partition the frozen sample IDs")

    selected_actions = []
    for path, row in action_by_path.items():
        decision = eligible_by_path.get(path)
        media_sha = str(row.get("sha256", ""))
        if decision is None:
            if excluded_by_path[path].get("sha256") != media_sha:
                raise ValueError(f"excluded media hash differs from action row: {path}")
            continue
        if (decision.get("sha256") != media_sha
                or decision.get("task") != row.get("task")
                or float(decision.get("query_time_log_seconds", float("nan")))
                != float(row.get("query_time_log_seconds", float("nan")))):
            raise ValueError(f"eligible media provenance differs from action row: {path}")
        selected_actions.append(row)

    state_by_path = {str(row.get("path")): row for row in state_rows}
    if len(state_by_path) != len(state_rows) or set(state_by_path) != set(all_paths):
        raise ValueError("state feature rows do not cover the original frozen sample IDs")
    for path, row in state_by_path.items():
        action = action_by_path[path]
        if (row.get("sha256") != action.get("sha256")
                or row.get("target") != action.get("target")
                or row.get("task_group") != action.get("task")):
            raise ValueError(f"state feature provenance differs from action row: {path}")
    state_indices = [index for index, row in enumerate(state_rows)
                     if str(row["path"]) in eligible_paths]
    if state_values.shape[0] != len(state_rows):
        raise ValueError("state feature tensor and examples differ before eligibility filtering")
    selected_states = [state_rows[index] for index in state_indices]
    selected_state_values = state_values[state_indices]

    text_paths = [str(value) for value in text_cache.get("sample_paths", [])]
    text_index = {path: index for index, path in enumerate(text_paths)}
    if len(text_index) != len(text_paths) or set(text_paths) != set(all_paths):
        raise ValueError("text feature rows do not cover the original frozen sample IDs")
    text_hashes = [str(value) for value in text_cache.get("media_sha256", [])]
    text_targets = [str(value) for value in text_cache.get("targets", [])]
    if len(text_hashes) != len(text_paths) or len(text_targets) != len(text_paths):
        raise ValueError("text feature media IDs, hashes and targets differ in length")
    for path, index in text_index.items():
        action = action_by_path[path]
        if (text_hashes[index] != action.get("sha256")
                or text_targets[index] != action.get("target")):
            raise ValueError(f"text feature provenance differs from action row: {path}")
    text_indices = [text_index[path] for path in text_paths if path in eligible_paths]
    selected_text_cache = {
        key: (value[text_indices] if isinstance(value, np.ndarray)
              and value.ndim > 0 and value.shape[0] == len(text_paths) else value)
        for key, value in text_cache.items()
    }

    video_by_path = {str(row.get("path")): row for row in video_rows}
    if len(video_by_path) != len(video_rows) or set(video_by_path) != eligible_paths:
        raise ValueError("video features do not cover exactly the eligible media IDs")
    if video_values.shape[0] != len(video_rows):
        raise ValueError("video feature tensor and examples differ before eligibility filtering")
    for path, row in video_by_path.items():
        action = action_by_path[path]
        eligibility_row = eligible_by_path[path]
        if (row.get("sha256") != action.get("sha256")
                or row.get("target") != action.get("target")
                or row.get("task") != action.get("task")
                or float(row.get("query_time_log_seconds", float("nan")))
                != float(action.get("query_time_log_seconds", float("nan")))):
            raise ValueError(f"video feature provenance differs from action row: {path}")
        recorded = row.get("selected_frame_times_log_seconds")
        expected_times = eligibility_row.get("selected_frame_times_log_seconds")
        if not isinstance(recorded, dict) or not isinstance(expected_times, dict):
            raise ValueError(f"video frame provenance is missing: {path}")
        for stream in VIDEO_STREAMS:
            if (stream not in recorded or stream not in expected_times
                    or not np.allclose(
                        recorded[stream], expected_times[stream], rtol=0, atol=1e-6
                    )):
                raise ValueError(f"video features differ from eligible frame selection: {path}")
    if (len(selected_actions) != int(eligibility.get("eligible_examples", -1))
            or len(excluded_paths) != int(eligibility.get("excluded_examples", -1))
            or len(action_rows) != int(eligibility.get("corpus_examples", -1))):
        raise ValueError("video-window eligibility counts differ from source rows")
    return (
        selected_actions, selected_states, video_rows, selected_text_cache,
        video_values, selected_state_values, len(excluded_paths),
    )


def _load_split(split: str, root: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    directory = root / f"{split}-joined-v1"
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite joined feature set: {directory}")
    source = root / f"{split}-corpus-v1"
    if split == "train":
        source = root / "train-corpus-v1"
        receipt_path = source / "train-corpus-receipt.json"
        action_path = source / "train-action-preflight-v2.json"
        state_dir = root / "train-state-text-v3"
        video_dir = root / "train-video-features-v3"
    else:
        receipt_path = source / f"{split}-corpus-receipt.json"
        action_path = root / f"{split}-action-preflight-v1.json"
        state_dir = root / f"{split}-state-text-v2"
        video_dir = root / (
            "calibration-video-features-v3"
            if split == "calibration" else "validation-video-features-v1"
        )
    state_result = json.loads((state_dir / "result.json").read_text(encoding="utf-8"))
    video_result = json.loads((video_dir / "result.json").read_text(encoding="utf-8"))
    text_dir = root / f"{split}-text-features-v2"
    text_result = json.loads((text_dir / "result.json").read_text(encoding="utf-8"))
    action = json.loads(action_path.read_text(encoding="utf-8"))
    state_examples = json.loads((state_dir / "examples.json").read_text(encoding="utf-8"))
    video_examples = json.loads((video_dir / "examples.json").read_text(encoding="utf-8"))
    video_values = np.load(video_dir / "video-features.npy", mmap_mode="r", allow_pickle=False)
    state_values = np.load(state_dir / "state-sequences.npy", mmap_mode="r", allow_pickle=False)
    with np.load(text_dir / "text-features.npz", allow_pickle=False) as cache:
        text_cache = {key: cache[key] for key in cache.files}
    if (state_result.get("split", split) != split
            or video_result.get("development_split") != (None if split == "train" else split)
            or text_result.get("split") != split
            or state_result.get("sealed_audit_opened") is not False
            or video_result.get("sealed_audit_opened") is not False
            or text_result.get("sealed_audit_opened") is not False):
        raise ValueError(f"feature provenance does not prove {split} isolation")
    manifest = json.loads(
        (ROOT / "manifests/pretrained_reuse/haptile-task-split-v1-20261122.json")
        .read_text(encoding="utf-8")
    )
    if (action.get("input_receipt_sha256") != sha256_file(receipt_path)
            or action.get("split_manifest_sha256")
            != sha256_file(
                ROOT / "manifests/pretrained_reuse/haptile-task-split-v1-20261122.json"
            )):
        raise ValueError(f"{split} target preflight provenance differs")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if split != "train" and receipt.get("accessed_splits") != [split]:
        raise ValueError(f"development receipt accessed another split: {split}")
    rows = action.get("samples")
    if not isinstance(rows, list):
        raise ValueError(f"missing target rows: {split}")
    eligibility_path = root / f"{split}-video-window-eligibility-v2.json"
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    rows, state_examples, video_examples, text_cache, video_values, state_values, excluded = (
        apply_video_window_eligibility(
            action_rows=rows,
            state_rows=state_examples,
            video_rows=video_examples,
            text_cache=text_cache,
            video_values=video_values,
            state_values=state_values,
            eligibility=eligibility,
            split_name=split,
            receipt_sha256=sha256_file(receipt_path),
            action_sha256=sha256_file(action_path),
            split_manifest_sha256=sha256_file(
                ROOT / "manifests/pretrained_reuse/haptile-task-split-v1-20261122.json"
            ),
        )
    )
    aligned = align_split_features(
        action_rows=rows,
        state_rows=state_examples,
        video_rows=video_examples,
        text_cache=text_cache,
        video_values=video_values,
        state_values=state_values,
    )
    expected_count = manifest["development_episode_counts"][split] - excluded
    if len(aligned["sample_paths"]) != expected_count:
        raise ValueError(f"eligible frozen split example count differs: {split}")
    directory.mkdir(parents=True)
    np.savez(directory / "decision-features.npz", **aligned)
    result = {
        "format": f"haptile_{split}_joined_decision_features_v1",
        "status": "complete_split_isolated_feature_join",
        "split": split,
        "examples": int(len(aligned["sample_paths"])),
        "pre_eligibility_examples": int(eligibility["corpus_examples"]),
        "video_window_excluded_examples": excluded,
        "video_window_eligibility_sha256": sha256_file(eligibility_path),
        "task_groups": int(len(set(aligned["task_groups"].tolist()))),
        "feature_shapes": {key: list(value.shape) for key, value in aligned.items()},
        "feature_file_sha256": sha256_file(directory / "decision-features.npz"),
        "input_hashes": {
            "receipt": sha256_file(receipt_path),
            "action_preflight": sha256_file(action_path),
            "state_result": sha256_file(state_dir / "result.json"),
            "video_result": sha256_file(video_dir / "result.json"),
            "text_result": sha256_file(text_dir / "result.json"),
            "video_window_eligibility": sha256_file(eligibility_path),
        },
        "sealed_audit_opened": False,
        "quantization_performed": False,
    }
    (directory / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return aligned, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    args = parser.parse_args()
    for split in args.splits:
        _, result = _load_split(split, args.artifact_root)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
