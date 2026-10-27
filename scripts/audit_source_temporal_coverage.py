"""Measure train-only cached audio-window and video-frame sequence coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

PLAN = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/perception-source-candidate-feature-plan-20261025-02/feature-plan.json"
)
PLAN_SHA = "9cf2fae738270a14e632b745b8a0f78214440eb057de76f0ba825d4413824247"


def summarize(records):
    windows, frames, padded, candidate_count = Counter(), Counter(), 0, 0
    total_windows, padded_windows, mostly_padded_windows = 0, 0, 0
    repeated_frame_records = 0
    for record in records:
        audio = record["audio_windows"]
        if not audio:
            raise ValueError("record has no audio windows")
        windows[len(audio)] += 1
        if any(
            w["source_end_us"] <= w["source_start_us"]
            or w["pad_left_us"] < 0
            or w["pad_right_us"] < 0
            for w in audio
        ):
            raise ValueError("invalid audio window")
        padded += int(any(w["pad_left_us"] or w["pad_right_us"] for w in audio))
        for window in audio:
            total_windows += 1
            pad = window["pad_left_us"] + window["pad_right_us"]
            actual = window["source_end_us"] - window["source_start_us"]
            if pad + actual != 2_000_000:
                raise ValueError("audio window must total exactly two seconds")
            padded_windows += int(pad > 0)
            mostly_padded_windows += int(pad > 1_000_000)
        ids = set()
        if not record["candidates"]:
            raise ValueError("record has no candidates")
        for candidate in record["candidates"]:
            candidate_count += 1
            frame_ids = [f["frame_id"] for f in candidate["frames"]]
            if not frame_ids or len(frame_ids) != len(set(frame_ids)):
                raise ValueError("missing or duplicate candidate frame IDs")
            if candidate["candidate_id"] in ids:
                raise ValueError("duplicate candidate ID")
            ids.add(candidate["candidate_id"])
            frames[len(frame_ids)] += 1
        repeated_frame_records += int(any(len(c["frames"]) == 1 for c in record["candidates"]))
    return {
        "records": len(records),
        "candidates": candidate_count,
        "audio_windows_per_record": dict(sorted(windows.items())),
        "video_frames_per_candidate": dict(sorted(frames.items())),
        "records_with_audio_padding": padded,
        "total_audio_windows": total_windows,
        "padded_audio_windows": padded_windows,
        "audio_windows_more_than_half_padded": mostly_padded_windows,
        "records_with_at_least_one_single_frame_candidate": repeated_frame_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output exists")
    raw = PLAN.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PLAN_SHA:
        raise ValueError("feature plan hash mismatch")
    plan = json.loads(raw)
    records = plan["train"]["records"]
    if len(records) != 2593 or len({row["video_id"] for row in records}) != 320:
        raise ValueError("unexpected train coverage")
    result = {
        "status": "PASS_TRAIN_ONLY_TEMPORAL_COVERAGE",
        "plan_sha256": PLAN_SHA,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summarize(records),
        "training": False,
        "model_run": False,
        "development_features_or_labels_read": False,
        "sealed_audit_opened": False,
        "limitations": "metadata counts only; no physical audio/video synchronization proof",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(result, target, indent=2)
        target.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
