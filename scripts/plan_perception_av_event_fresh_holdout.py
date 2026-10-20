"""Plan one fixed-checkpoint evaluation on previously unused TRAIN videos.

This is a development holdout, not a blind audit: the source annotations were
already parsed during feasibility screening. No model outputs are consulted.
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

PLAN_SHA256 = "9ed4f44f3a0cc85465c5a2955706a957326ef36694311c25b4ee51e3c9f18a25"
TRAIN_MEDIA_MANIFEST_SHA256 = "5ac15ab7ef7efda29a803f783b4a3a13fd3895ebaecbdf9c9c7e0282b9195918"
VALID_MEDIA_MANIFEST_SHA256 = "26d941165f30922d17a528225a004d7eaa1522f5ca557a16f8512e319f0fb6b4"
HOLDOUT_VIDEO_COUNT = 200


def _sha256_order(video_ids: set[str]) -> list[str]:
    return sorted(
        video_ids,
        key=lambda video_id: hashlib.sha256(
            f"perception-av-event-v1:fresh-train-eval:17:{video_id}".encode()
        ).digest(),
    )


def select_holdout_ids(
    candidates: set[str], previously_used: set[str], count: int = HOLDOUT_VIDEO_COUNT
) -> list[str]:
    if count < 1:
        raise ValueError("holdout count must be positive")
    available = candidates - previously_used
    selected = _sha256_order(available)[:count]
    if len(selected) != count:
        raise ValueError(f"need {count} fresh TRAIN videos; found {len(selected)}")
    if len(set(selected)) != count or set(selected) & previously_used:
        raise ValueError("holdout video grouping is invalid")
    return selected


def create_plan(root: Path = ROOT, count: int = HOLDOUT_VIDEO_COUNT) -> dict:
    root = Path(root)
    feasibility_path = root / "perception-av-event-task-feasibility-01" / "result.json"
    if sha256_file(feasibility_path) != FEASIBILITY_SHA256:
        raise ValueError("pinned source-feasibility receipt changed")
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    media_plan_path = root / "perception-av-event-media-plan-02" / "plan.json"
    if sha256_file(media_plan_path) != PLAN_SHA256:
        raise ValueError("pinned original media plan changed")
    media_plan = json.loads(media_plan_path.read_text(encoding="utf-8"))
    current_train_manifest = root / "perception-av-event-train-media-01" / "manifest.json"
    current_valid_manifest = root / "perception-av-event-valid-media-01" / "manifest.json"
    if sha256_file(current_train_manifest) != TRAIN_MEDIA_MANIFEST_SHA256:
        raise ValueError("original TRAIN acquisition manifest changed")
    if sha256_file(current_valid_manifest) != VALID_MEDIA_MANIFEST_SHA256:
        raise ValueError("original VALID acquisition manifest changed")

    prior_ids, prior_pins = read_prior_media_ids(root)
    prior_ids |= {
        row["video_id"]
        for split in ("train", "valid")
        for row in media_plan["splits"][split]["rows"]
    }
    train_source = root / "perception-test-source-audit-01"
    annotations, cuts, source_hashes = load_annotations(
        train_source / "train_annotations.zip",
        train_source / "cut_frame_mapping_train.json",
        "train",
        {
            "annotations": ANNOTATION_SHA256["train_annotations.zip"],
            "cuts": ANNOTATION_SHA256["cut_frame_mapping_train.json"],
        },
    )
    summary = summarize_split(annotations, cuts, "train")
    candidates = set(summary["candidate_video_ids"])
    selected = select_holdout_ids(candidates, prior_ids, count)
    rows = archive_rows(TRAIN_ARCHIVE_PIN, selected, annotations, cuts, "train")
    event_counts = {video_id: 0 for video_id in selected}
    for event in summary["candidate_records"]:
        if event["video_id"] in event_counts:
            event_counts[event["video_id"]] += 1
    for row in rows:
        row["candidate_events"] = event_counts[row["video_id"]]
    total_bytes = sum(row["bytes"] for row in rows)
    if len(rows) != count or total_bytes > MAX_SPLIT_BYTES:
        raise ValueError("holdout media inventory exceeds the fixed plan bounds")
    return {
        "status": "PASS_FRESH_TRAIN_HOLDOUT_PLAN",
        "purpose": "single fixed-checkpoint inference comparison; no fitting or selection",
        "task_id": "perception-av-event-source-selection-v1-fresh-holdout",
        "seed": 17,
        "source_revision": "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf",
        "source_commit": "809c0f8226a780039a89c8066709d15d39a53f8a",
        "source_annotation_sha256": source_hashes,
        "feasibility_sha256": FEASIBILITY_SHA256,
        "original_media_plan_sha256": PLAN_SHA256,
        "original_acquisition_manifest_sha256": {
            "train": TRAIN_MEDIA_MANIFEST_SHA256,
            "valid": VALID_MEDIA_MANIFEST_SHA256,
        },
        "earlier_media_manifest_sha256": prior_pins,
        "excluded_video_ids": len(prior_ids),
        "eligible_unseen_video_ids": len(candidates - prior_ids),
        "selection_rule": (
            "first 200 SHA-256 ordered IDs by "
            "perception-av-event-v1:fresh-train-eval:17:{video_id}"
        ),
        "official_split": (
            "TRAIN only; annotations previously parsed for feasibility, so not a blind audit"
        ),
        "participant_disjointness": "UNKNOWN; public annotations do not expose participant IDs",
        "model_inputs_exclude": [
            "sound label",
            "source parent ID",
            "ground-truth candidate track IDs",
            "video ID",
            "filename",
            "split ID",
        ],
        "archive_pin": TRAIN_ARCHIVE_PIN,
        "candidate_rule": feasibility["candidate_rule"],
        "video_ids_sha256": hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest(),
        "candidate_videos": len(rows),
        "candidate_events": sum(row["candidate_events"] for row in rows),
        "total_uncompressed_bytes": total_bytes,
        "max_clip_bytes": MAX_CLIP_BYTES,
        "rows": rows,
        "training": False,
        "checkpoint_selection": False,
        "test_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output directory required")
    plan = create_plan()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "plan.json"
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        key: plan[key]
        for key in (
            "candidate_videos",
            "candidate_events",
            "total_uncompressed_bytes",
            "video_ids_sha256",
        )
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
