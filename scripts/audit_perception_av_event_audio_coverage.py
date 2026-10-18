"""Estimate whether every selected event window overlaps its pinned clip audio."""

import argparse
import json
from pathlib import Path

from scripts.audit_perception_av_correspondence import sha256_file
from scripts.audit_perception_av_event_task import (
    TRAIN_SOURCE_SHA256,
    VALID_SOURCE_SHA256,
    load_annotations,
    summarize_split,
)

PLAN_SHA256 = "9ed4f44f3a0cc85465c5a2955706a957326ef36694311c25b4ee51e3c9f18a25"


def estimate_audio_coverage(frame_interval, frame_rate, audio_samples, sample_rate):
    if (
        len(frame_interval) != 2
        or any(type(value) is not int or value < 0 for value in frame_interval)
        or frame_interval[1] < frame_interval[0]
        or frame_rate <= 0
        or audio_samples < 0
        or sample_rate <= 0
    ):
        raise ValueError("invalid event interval or media timing metadata")
    start = frame_interval[0] / frame_rate
    end = (frame_interval[1] + 1) / frame_rate
    audio_end = audio_samples / sample_rate
    available = max(0.0, min(end, audio_end) - start)
    requested = end - start
    return {
        "requested_seconds": requested,
        "available_seconds": available,
        "missing_tail_seconds": max(0.0, end - audio_end),
        "coverage_fraction": available / requested,
    }


def audit(plan_path, train_root, valid_root):
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("media plan hash changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS_FRESH_EVENT_MEDIA_PLAN" or plan.get("test_opened"):
        raise ValueError("expected pinned train/valid-only media plan")
    roots = {"train": Path(train_root), "valid": Path(valid_root)}
    source_pins = {"train": TRAIN_SOURCE_SHA256, "valid": VALID_SOURCE_SHA256}
    output = {
        "status": "PASS_EVENT_AUDIO_WINDOW_COVERAGE_ESTIMATE",
        "plan_sha256": PLAN_SHA256,
        "method": (
            "inclusive annotation frames mapped by pinned metadata frame_rate; audio duration "
            "estimated from audio_samples/sample_rate; refine with decoded PTS before features"
        ),
        "test_opened": False,
        "model_loaded": False,
        "training": False,
        "gpu_used": False,
        "splits": {},
    }
    for split, root in roots.items():
        annotation_name = f"{split}_annotations.zip"
        cut_name = f"cut_frame_mapping_{split}.json"
        data, cuts, _ = load_annotations(
            root / annotation_name,
            root / cut_name,
            split,
            {
                "annotations": source_pins[split]["annotations"],
                "cuts": source_pins[split]["cuts"],
            },
        )
        selected_ids = {row["video_id"] for row in plan["splits"][split]["rows"]}
        selected_rows = {
            row["video_id"]: row for row in plan["splits"][split]["rows"]
        }
        candidate_records = summarize_split(data, cuts, split)["candidate_records"]
        selected = [row for row in candidate_records if row["video_id"] in selected_ids]
        if not selected:
            raise ValueError(f"no selected event records in {split}")
        events = []
        for event in selected:
            media = selected_rows[event["video_id"]]["metadata"]
            coverage = estimate_audio_coverage(
                event["frame_interval"],
                media["frame_rate"],
                media["audio_samples"],
                media["audio_sample_rate"],
            )
            events.append(
                {
                    "video_id": event["video_id"],
                    "sound_id": event["sound_id"],
                    "frame_interval": event["frame_interval"],
                    **coverage,
                }
            )
        output["splits"][split] = {
            "candidate_events": len(events),
            "fully_covered_by_metadata_estimate": sum(
                event["missing_tail_seconds"] == 0 for event in events
            ),
            "partial_tail_by_metadata_estimate": sum(
                event["missing_tail_seconds"] > 0 for event in events
            ),
            "zero_overlap": sum(event["available_seconds"] == 0 for event in events),
            "minimum_coverage_fraction": min(
                event["coverage_fraction"] for event in events
            ),
            "maximum_missing_tail_seconds": max(
                event["missing_tail_seconds"] for event in events
            ),
            "events": events,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--valid-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output path required")
    result = audit(args.plan, args.train_source, args.valid_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                split: {
                    key: value
                    for key, value in values.items()
                    if key != "events"
                }
                for split, values in result["splits"].items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
