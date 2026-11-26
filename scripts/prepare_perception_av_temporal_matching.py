"""Prepare a video-grouped audio/video temporal-matching task manifest.

The manifest uses only official train and validation annotations. It does not
load the official test split or media bytes. Output media acquisition and model
inference are separate, later steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_REVISION = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
SEED = 17
ALLOWED_LABEL_IDS = (0, 1, 4, 5, 6, 7, 8, 9, 10, 11)
WINDOW_SECONDS = 2.0
MIN_CENTER_GAP_SECONDS = 2.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(*parts: object) -> str:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_annotation_archive(path: Path) -> dict[str, dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        json_members = [name for name in archive.namelist() if name.endswith(".json")]
        if len(json_members) != 1:
            raise ValueError(f"expected one annotation JSON in {path}, found {len(json_members)}")
        value = json.loads(archive.read(json_members[0]))
    if not isinstance(value, dict):
        raise ValueError(f"annotation root in {path} must be an object")
    return value


def load_fixed_events(path: Path, *, expected_split: str) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("official_test_split_opened") is True or value.get("test_split_opened") is True:
        raise ValueError(f"test split must remain unopened: {path}")
    rows = value.get("splits", {}).get(expected_split)
    if rows is None and expected_split == "valid":
        rows = value.get("events")
    if not isinstance(rows, list):
        raise ValueError(f"missing {expected_split} event rows in {path}")
    by_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("split", expected_split) != expected_split:
            raise ValueError(f"unexpected split row in {path}: {row.get('split')}")
        video_id = str(row["video_id"])
        if video_id in by_video:
            raise ValueError(f"duplicate video_id in selected events: {video_id}")
        by_video[video_id] = row
    return by_video


def event_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = record["metadata"]
    duration = min(
        float(metadata["audio_samples"]) / float(metadata["audio_sample_rate"]),
        float(metadata["num_frames"]) / float(metadata["frame_rate"]),
    )
    rows = []
    for event in record.get("sound_localisation", []):
        label_id = int(event.get("label_id", -1))
        if label_id not in ALLOWED_LABEL_IDS:
            continue
        start, end = (float(value) / 1_000_000 for value in event["timestamps"])
        center = (start + end) / 2
        half_window = WINDOW_SECONDS / 2
        if center - half_window < 0 or center + half_window > duration:
            continue
        rows.append(
            {
                "event_id": int(event["id"]),
                "label_id": label_id,
                "label": str(event["label"]).strip(),
                "start_s": start,
                "end_s": end,
                "center_s": center,
                "window_s": [center - half_window, center + half_window],
            }
        )
    return rows


def eligible_pairs(
    video_id: str,
    record: dict[str, Any],
    *,
    anchor_event_id: int | None = None,
) -> list[dict[str, Any]]:
    rows = event_rows(record)
    anchors = [row for row in rows if anchor_event_id is None or row["event_id"] == anchor_event_id]
    pairs = []
    for anchor in anchors:
        for partner in rows:
            if partner["event_id"] == anchor["event_id"]:
                continue
            if partner["label_id"] != anchor["label_id"]:
                continue
            gap = abs(partner["center_s"] - anchor["center_s"])
            if gap < MIN_CENTER_GAP_SECONDS:
                continue
            pairs.append(
                {
                    "video_id": video_id,
                    "label_id": anchor["label_id"],
                    "label": anchor["label"],
                    "anchor": anchor,
                    "partner": partner,
                    "center_gap_s": gap,
                }
            )
    return pairs


def choose_pair(
    video_id: str,
    record: dict[str, Any],
    *,
    anchor_event_id: int | None = None,
    split: str,
) -> dict[str, Any] | None:
    pairs = eligible_pairs(video_id, record, anchor_event_id=anchor_event_id)
    if not pairs:
        return None
    return min(
        pairs,
        key=lambda pair: stable_rank(
            SEED,
            "perception-av-temporal-match-v1",
            split,
            video_id,
            pair["anchor"]["event_id"],
            pair["partner"]["event_id"],
        ),
    )


def _make_trial(pair: dict[str, Any], *, split: str) -> dict[str, Any]:
    video_id = pair["video_id"]
    anchor_id = pair["anchor"]["event_id"]
    partner_id = pair["partner"]["event_id"]
    order = stable_rank(SEED, "perception-av-temporal-option-order-v1", split, video_id)
    options = [
        {
            "option_id": f"audio_{anchor_id}",
            "event_id": anchor_id,
            "audio_window_s": pair["anchor"]["window_s"],
            "role": "synchronized",
        },
        {
            "option_id": f"audio_{partner_id}",
            "event_id": partner_id,
            "audio_window_s": pair["partner"]["window_s"],
            "role": "same_video_same_label_other_time",
        },
    ]
    if int(order[:8], 16) % 2:
        options.reverse()
    target = next(index for index, option in enumerate(options) if option["role"] == "synchronized")
    return {
        "sample_id": f"{split}:{video_id}:{anchor_id}:{partner_id}",
        "split": split,
        "video_id": video_id,
        "group_id": video_id,
        "question": "Which candidate audio window is synchronized with this video interval?",
        "label_id": pair["label_id"],
        "label_name": pair["label"],
        "video_event_id": anchor_id,
        "video_window_s": pair["anchor"]["window_s"],
        "video_frame_times_s": [
            pair["anchor"]["window_s"][0] + index * WINDOW_SECONDS / 7 for index in range(8)
        ],
        "options": options,
        "target_index": target,
        "same_label_hard_negative": True,
        "same_video_hard_negative": True,
        "audio_center_gap_s": pair["center_gap_s"],
        "source_video_id": video_id,
    }


def _select_additional_train(
    train_annotations: dict[str, dict[str, Any]],
    *,
    excluded_video_ids: set[str],
    count: int,
) -> list[dict[str, Any]]:
    by_label: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for video_id, record in train_annotations.items():
        if video_id in excluded_video_ids:
            continue
        pair = choose_pair(video_id, record, split="train")
        if pair is None:
            continue
        rank = stable_rank(SEED, "perception-av-temporal-train-pool-v1", video_id)
        by_label[pair["label_id"]].append((rank, pair))
    for label_id in by_label:
        by_label[label_id].sort(key=lambda item: item[0])

    selected = []
    labels = sorted(by_label)
    while labels and len(selected) < count:
        remaining = []
        for label_id in labels:
            candidates = by_label[label_id]
            if candidates and len(selected) < count:
                selected.append(candidates.pop(0)[1])
            if candidates:
                remaining.append(label_id)
        labels = remaining
    if len(selected) != count:
        raise ValueError(f"requested {count} added train videos, found only {len(selected)}")
    return selected


def build_manifest(
    *,
    train_annotations_path: Path,
    valid_annotations_path: Path,
    fixed_cohort_path: Path,
    new_valid_selection_path: Path,
    added_train_videos: int = 100,
) -> dict[str, Any]:
    train_annotations = load_annotation_archive(train_annotations_path)
    valid_annotations = load_annotation_archive(valid_annotations_path)
    fixed_train = load_fixed_events(fixed_cohort_path, expected_split="train")
    fixed_valid = load_fixed_events(fixed_cohort_path, expected_split="valid")
    fresh_valid = load_fixed_events(new_valid_selection_path, expected_split="valid")
    if set(fixed_train) & set(fixed_valid) or set(fixed_train) & set(fresh_valid):
        raise ValueError("train/validation video identity overlap")
    if set(fixed_valid) & set(fresh_valid):
        raise ValueError("fresh validation selection overlaps fixed validation")

    train_rows = []
    for video_id, fixed in fixed_train.items():
        record = train_annotations.get(video_id)
        if record is None:
            raise ValueError(
                f"fixed train video missing from official train annotations: {video_id}"
            )
        pair = choose_pair(video_id, record, anchor_event_id=int(fixed["event_id"]), split="train")
        if pair is not None:
            train_rows.append(_make_trial(pair, split="train"))

    excluded = set(fixed_train) | set(fixed_valid) | set(fresh_valid)
    additions = _select_additional_train(
        train_annotations,
        excluded_video_ids=excluded,
        count=added_train_videos,
    )
    train_rows.extend(_make_trial(pair, split="train") for pair in additions)

    valid_rows = []
    for selected in (fixed_valid, fresh_valid):
        for video_id, fixed in selected.items():
            record = valid_annotations.get(video_id)
            if record is None:
                raise ValueError(
                    f"selected validation video missing from official validation: {video_id}"
                )
            pair = choose_pair(
                video_id, record, anchor_event_id=int(fixed["event_id"]), split="valid"
            )
            if pair is not None:
                valid_rows.append(_make_trial(pair, split="valid"))

    train_ids = [row["video_id"] for row in train_rows]
    valid_ids = [row["video_id"] for row in valid_rows]
    if len(train_ids) != len(set(train_ids)) or len(valid_ids) != len(set(valid_ids)):
        raise ValueError("duplicate video groups in prepared task rows")
    if set(train_ids) & set(valid_ids):
        raise ValueError("prepared train/validation video identity overlap")

    return {
        "schema_version": 1,
        "status": "PASS_AV_TEMPORAL_MATCHING_MANIFEST_PREFLIGHT",
        "task_id": "perception-av-same-video-same-label-temporal-match-v1",
        "task_claim": (
            "Exploratory audio/video matching task: choose which same-video, "
            "same-label audio window matches an 8-frame video interval. "
            "Not general AV understanding."
        ),
        "source_revision": SOURCE_REVISION,
        "seed": SEED,
        "label_ids": list(ALLOWED_LABEL_IDS),
        "window_seconds": WINDOW_SECONDS,
        "minimum_same_video_same_label_audio_center_gap_seconds": MIN_CENTER_GAP_SECONDS,
        "video_frame_count": 8,
        "split_policy": "group by source video_id; no video identity crosses train and validation",
        "selection_inputs_sha256": {
            "train_annotations": sha256_file(train_annotations_path),
            "validation_annotations": sha256_file(valid_annotations_path),
            "fixed_cohort": sha256_file(fixed_cohort_path),
            "fresh_validation_selection": sha256_file(new_valid_selection_path),
        },
        "train": train_rows,
        "validation": valid_rows,
        "summary": {
            "train_groups": len(train_rows),
            "validation_groups": len(valid_rows),
            "train_by_label": dict(Counter(str(row["label_id"]) for row in train_rows)),
            "validation_by_label": dict(Counter(str(row["label_id"]) for row in valid_rows)),
            "train_validation_video_overlap": 0,
            "sealed_audit_loaded": False,
            "official_test_split_opened": False,
            "validation_scope": (
                "development-only; prior source and adjacent labels were previously inspected"
            ),
            "added_train_video_count": added_train_videos,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--valid-annotations", type=Path, required=True)
    parser.add_argument("--fixed-cohort", type=Path, required=True)
    parser.add_argument("--fresh-valid-selection", type=Path, required=True)
    parser.add_argument("--added-train-videos", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    result = build_manifest(
        train_annotations_path=args.train_annotations,
        valid_annotations_path=args.valid_annotations,
        fixed_cohort_path=args.fixed_cohort,
        new_valid_selection_path=args.fresh_valid_selection,
        added_train_videos=args.added_train_videos,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"output": str(args.output), **result["summary"]}, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
