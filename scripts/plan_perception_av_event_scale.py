"""Freeze a scene-disjoint scale experiment from untouched Perception TRAIN videos.

Source annotations were parsed in earlier feasibility work. This is a development
split, not a blind audit. The 200-video fixed-checkpoint evaluation cohort is
explicitly excluded from model fitting and checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_task import load_annotations, summarize_split
from scripts.plan_perception_av_event_media import (
    ANNOTATION_SHA256,
    FEASIBILITY_SHA256,
    MAX_CLIP_BYTES,
    MAX_SPLIT_BYTES,
    ROOT,
    TRAIN_ARCHIVE_PIN,
    archive_rows,
    read_prior_media_ids,
)

FRESH_HOLDOUT_PLAN_SHA256 = "887a1eea4b5a7b82d441025842465a09d0ffb33c97787073042ad1a1680f8eb3"
FRESH_HOLDOUT_EVAL_SHA256 = "19fcae15cfe7ac5d385cf2d94cde4fe80468f61b7b1b90ec7b6ba5b8bda96fab"
FRESH_HOLDOUT_JOINT_PREDICTIONS_SHA256 = (
    "ce1132d8cd9420931c94aeed9fd97ec6e0fe9e75da496071b1bd0af2fe2f8642"
)
FRESH_HOLDOUT_VIDEO_ONLY_PREDICTIONS_SHA256 = (
    "4acac4d853302a369aca1aabbe3368bf923a0a0ee41fcf50a4a8a74f3a745a71"
)
FRESH_HOLDOUT_VIDEO_COUNT = 200
ORIGINAL_MEDIA_PLAN_SHA256 = "9ed4f44f3a0cc85465c5a2955706a957326ef36694311c25b4ee51e3c9f18a25"
SCALE_VIDEO_COUNT = 388
SEED = 17
SPLIT_FRACTIONS = (
    ("fit", 60),
    ("selection", 15),
    ("calibration", 10),
    ("development_validation", 15),
)


def _ordered(video_ids: set[str], purpose: str) -> list[str]:
    return sorted(
        video_ids,
        key=lambda video_id: hashlib.sha256(
            f"perception-av-event-scale-v1:{SEED}:{purpose}:{video_id}".encode()
        ).digest(),
    )


def partition_videos(video_ids: set[str]) -> dict[str, list[str]]:
    if len(video_ids) != SCALE_VIDEO_COUNT:
        raise ValueError(f"expected {SCALE_VIDEO_COUNT} unused video IDs, found {len(video_ids)}")
    ordered = _ordered(video_ids, "split")
    exact = {name: len(ordered) * percent / 100 for name, percent in SPLIT_FRACTIONS}
    sizes = {name: int(value) for name, value in exact.items()}
    remainder = len(ordered) - sum(sizes.values())
    remainder_order = sorted(
        exact,
        key=lambda name: (
            exact[name] - sizes[name],
            -[key for key, _ in SPLIT_FRACTIONS].index(name),
        ),
        reverse=True,
    )
    for name in remainder_order[:remainder]:
        sizes[name] += 1
    result = {}
    cursor = 0
    for name, _ in SPLIT_FRACTIONS:
        result[name] = ordered[cursor : cursor + sizes[name]]
        cursor += sizes[name]
    if cursor != len(ordered) or set().union(*(set(x) for x in result.values())) != video_ids:
        raise ValueError("video-level split partition is incomplete")
    if any(
        set(result[a]) & set(result[b]) for i, a in enumerate(result) for b in list(result)[i + 1 :]
    ):
        raise ValueError("video-level splits overlap")
    return result


def select_unused_video_ids(
    candidate_ids: set[str], prior_ids: set[str], evaluated_ids: set[str]
) -> set[str]:
    eligible = candidate_ids - prior_ids
    if not evaluated_ids <= eligible:
        raise ValueError("fixed-checkpoint evaluation IDs are missing from the eligible inventory")
    remaining = eligible - evaluated_ids
    if len(eligible) != 588 or len(evaluated_ids) != FRESH_HOLDOUT_VIDEO_COUNT:
        raise ValueError("pinned unused/evaluated video inventory changed")
    if len(remaining) != SCALE_VIDEO_COUNT:
        raise ValueError(f"expected {SCALE_VIDEO_COUNT} unused IDs, found {len(remaining)}")
    return remaining


def create_plan(root: Path = ROOT) -> dict:
    root = Path(root)
    feasibility_path = root / "perception-av-event-task-feasibility-01" / "result.json"
    if sha256_file(feasibility_path) != FEASIBILITY_SHA256:
        raise ValueError("pinned feasibility receipt changed")
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))

    prior_ids, prior_pins = read_prior_media_ids(root)
    original_plan_path = root / "perception-av-event-media-plan-02" / "plan.json"
    if sha256_file(original_plan_path) != ORIGINAL_MEDIA_PLAN_SHA256:
        raise ValueError("original pinned media plan changed")
    original_plan = json.loads(original_plan_path.read_text(encoding="utf-8"))
    prior_ids |= {
        row["video_id"]
        for split in ("train", "valid")
        for row in original_plan["splits"][split]["rows"]
    }
    holdout_plan_path = root / "perception-av-event-fresh-holdout-plan-01" / "plan.json"
    if sha256_file(holdout_plan_path) != FRESH_HOLDOUT_PLAN_SHA256:
        raise ValueError("fixed-checkpoint holdout plan changed")
    holdout_plan = json.loads(holdout_plan_path.read_text(encoding="utf-8"))
    holdout_ids = {row["video_id"] for row in holdout_plan["rows"]}
    if len(holdout_ids) != FRESH_HOLDOUT_VIDEO_COUNT:
        raise ValueError("fixed-checkpoint holdout video count changed")
    eval_dir = root / "perception-av-event-fresh-holdout-eval-01"
    eval_result_path = eval_dir / "result.json"
    if sha256_file(eval_result_path) != FRESH_HOLDOUT_EVAL_SHA256:
        raise ValueError("fixed-checkpoint holdout evaluation receipt changed")
    eval_result = json.loads(eval_result_path.read_text(encoding="utf-8"))
    prediction_pins = {
        "joint-predictions.jsonl": FRESH_HOLDOUT_JOINT_PREDICTIONS_SHA256,
        "video_only_matched-predictions.jsonl": FRESH_HOLDOUT_VIDEO_ONLY_PREDICTIONS_SHA256,
    }
    if (
        eval_result.get("status") != "PASS_FRESH_FIXED_CHECKPOINT_EVALUATION"
        or eval_result.get("events") != 504
    ):
        raise ValueError("fixed-checkpoint evaluation is not complete")
    if (
        eval_result["models"]["joint"]["predictions_sha256"]
        != FRESH_HOLDOUT_JOINT_PREDICTIONS_SHA256
        or eval_result["models"]["video_only_matched"]["predictions_sha256"]
        != FRESH_HOLDOUT_VIDEO_ONLY_PREDICTIONS_SHA256
    ):
        raise ValueError("fixed-checkpoint evaluation receipt does not pin prediction files")
    for filename, expected_sha256 in prediction_pins.items():
        path = eval_dir / filename
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"fixed-checkpoint predictions changed: {filename}")
        predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        ids = {row["video_id"] for row in predictions}
        if len(predictions) != 504 or ids != holdout_ids:
            raise ValueError(f"fixed-checkpoint evaluation coverage changed: {filename}")

    source_dir = root / "perception-test-source-audit-01"
    annotations, cuts, source_hashes = load_annotations(
        source_dir / "train_annotations.zip",
        source_dir / "cut_frame_mapping_train.json",
        "train",
        {
            "annotations": ANNOTATION_SHA256["train_annotations.zip"],
            "cuts": ANNOTATION_SHA256["cut_frame_mapping_train.json"],
        },
    )
    summary = summarize_split(annotations, cuts, "train")
    if summary["candidate_video_ids"] != feasibility["train"]["candidate_video_ids"]:
        raise ValueError("recomputed eligible IDs differ from pinned feasibility receipt")

    remaining = select_unused_video_ids(set(summary["candidate_video_ids"]), prior_ids, holdout_ids)
    split_ids = partition_videos(remaining)
    video_to_split = {video_id: split for split, ids in split_ids.items() for video_id in ids}
    rows = archive_rows(TRAIN_ARCHIVE_PIN, sorted(remaining), annotations, cuts, "train")

    events_by_video = {video_id: 0 for video_id in remaining}
    for event in summary["candidate_records"]:
        if event["video_id"] in events_by_video:
            events_by_video[event["video_id"]] += 1
    for row in rows:
        row["model_split"] = video_to_split[row["video_id"]]
        row["candidate_events"] = events_by_video[row["video_id"]]

    split_summary = {}
    for split, ids in split_ids.items():
        split_rows = [row for row in rows if row["model_split"] == split]
        split_summary[split] = {
            "video_count": len(ids),
            "video_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
            "candidate_events": sum(row["candidate_events"] for row in split_rows),
            "total_uncompressed_bytes": sum(row["bytes"] for row in split_rows),
        }
    total_bytes = sum(row["bytes"] for row in rows)
    if total_bytes > MAX_SPLIT_BYTES or any(row["bytes"] > MAX_CLIP_BYTES for row in rows):
        raise ValueError("fresh source media exceeds pinned acquisition limits")

    return {
        "status": "PASS_FRESH_VIDEO_SCALE_PLAN",
        "task_id": "perception-av-event-source-selection-v1-scale",
        "seed": SEED,
        "source_commit": "77c2949c6d454ad81ba335f2a0b2a1601bbb8297",
        "source_annotation_sha256": source_hashes,
        "feasibility_sha256": FEASIBILITY_SHA256,
        "prior_media_manifest_sha256": prior_pins,
        "original_media_plan_sha256": ORIGINAL_MEDIA_PLAN_SHA256,
        "fixed_checkpoint_holdout_plan_sha256": FRESH_HOLDOUT_PLAN_SHA256,
        "fixed_checkpoint_holdout_eval_sha256": FRESH_HOLDOUT_EVAL_SHA256,
        "fixed_checkpoint_holdout_prediction_sha256": prediction_pins,
        "fixed_checkpoint_holdout_video_ids_sha256": holdout_plan["video_ids_sha256"],
        "fixed_checkpoint_holdout_video_count_excluded": len(holdout_ids),
        "participant_disjointness": "UNKNOWN; participant IDs are absent from public annotations",
        "annotation_exposure": (
            "all source TRAIN annotations were previously parsed for feasibility"
        ),
        "archive_pin": TRAIN_ARCHIVE_PIN,
        "split_rule": "SHA-256 ordered video IDs by fixed seed; no event-level splitting",
        "split_percentages": dict(SPLIT_FRACTIONS),
        "split_summary": split_summary,
        "candidate_videos": len(rows),
        "candidate_events": sum(row["candidate_events"] for row in rows),
        "total_uncompressed_bytes": total_bytes,
        "required_free_bytes_with_1_2x_margin": int(total_bytes * 1.2),
        "training": False,
        "test_opened": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output directory required")
    plan = create_plan()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: plan[key]
                for key in (
                    "candidate_videos",
                    "candidate_events",
                    "total_uncompressed_bytes",
                    "required_free_bytes_with_1_2x_margin",
                    "split_summary",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
