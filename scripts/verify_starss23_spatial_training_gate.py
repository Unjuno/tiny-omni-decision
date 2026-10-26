"""Verify one fixed dev-train window's timestamps, labels, and spatial-token map."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.preflight_starss23_spatial_decision import (
    EVENT_CLASSES,
    EVENT_NAMES,
    METADATA_PATH,
    METADATA_SHA256,
    SECTORS,
    SPLIT_PATH,
    SPLIT_SHA256,
    interval_sector_targets,
)

WINDOW_RESULT = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-window-smoke-01/result.json"
)
WINDOW_ID = "fold3_room12_mix001:t0000-0005"
WINDOW_PATH = Path(
    "C:/CodexArtifacts/pretrained-reuse-r2/starss23-window-smoke-01/fold3_room12_mix001-0000.npz"
)
WINDOW_SHA256 = "05f4be05373f280c4cbff0e0cb1d4ed9d50d33752cf97b7ff04376954e616c55"


def verify_training_window() -> dict:
    if sha256_file(SPLIT_PATH) != SPLIT_SHA256:
        raise ValueError("pinned training split changed")
    if sha256_file(METADATA_PATH) != METADATA_SHA256:
        raise ValueError("pinned metadata archive changed")
    if sha256_file(WINDOW_PATH) != WINDOW_SHA256:
        raise ValueError("pinned dev-train window changed")

    split_rows = [
        json.loads(line)
        for line in SPLIT_PATH.read_text(encoding="utf-8").splitlines()
    ]
    clip_rows = [row for row in split_rows if row.get("clip") == "fold3_room12_mix001"]
    if len(clip_rows) != 1 or clip_rows[0].get("split") != "train":
        raise ValueError("fixed window recording is not in the pinned training split")
    metadata_member = clip_rows[0]["member"]

    window_report = json.loads(WINDOW_RESULT.read_text(encoding="utf-8"))
    windows = [row for row in window_report["windows"] if row.get("id") == WINDOW_ID]
    if len(windows) != 1 or windows[0].get("split") != "dev-train":
        raise ValueError("fixed window is not a dev-train window")
    window = windows[0]
    start_seconds = float(window["start_seconds"])
    end_seconds = float(window["end_seconds"])
    frame_times = [float(value) for value in window["video_seconds"]]
    if not frame_times or any(not start_seconds <= value < end_seconds for value in frame_times):
        raise ValueError("video frame presentation times do not lie inside the audio window")
    if len(frame_times) != 8 or window["audio_shape"] != [120000, 4]:
        raise ValueError("fixed audio/video window geometry changed")

    with zipfile.ZipFile(METADATA_PATH) as archive:
        if not metadata_member.startswith("metadata_dev/dev-train-"):
            raise ValueError("annotation member is outside dev-train")
        rows = list(csv.reader(io.StringIO(archive.read(metadata_member).decode("utf-8"))))
    start_tick = round(start_seconds * 10)
    stop_tick = round(end_seconds * 10)
    targets = interval_sector_targets(rows, start_tick, stop_tick)
    expected_targets = len(EVENT_CLASSES) * len(SECTORS)
    if len(targets) != expected_targets:
        raise ValueError("wrong number of event-sector targets")

    target_rows = [
        {
            "event_class": event_class,
            "event": EVENT_NAMES[event_class],
            "sector": SECTORS[sector],
            "options": ["no", "yes"],
            "target": targets[event_index * len(SECTORS) + sector],
        }
        for event_index, event_class in enumerate(EVENT_CLASSES)
        for sector in range(len(SECTORS))
    ]
    return {
        "status": "PASS",
        "scope": "one pinned dev-train five-second window; no held-out labels",
        "window_id": WINDOW_ID,
        "window_sha256": WINDOW_SHA256,
        "window_manifest_sha256": sha256_file(WINDOW_RESULT),
        "training_split_sha256": SPLIT_SHA256,
        "metadata_archive_sha256": METADATA_SHA256,
        "metadata_member": metadata_member,
        "window_seconds": [start_seconds, end_seconds],
        "annotation_tick_interval": [start_tick, stop_tick],
        "video_presentation_seconds": frame_times,
        "timestamp_alignment": (
            "all eight video presentation times fall inside the paired "
            "five-second audio interval"
        ),
        "azimuth_convention": (
            "STARSS: 0 degrees front, +90 degrees left; panorama x maps "
            "+180 at left, 0 at center, -180 at right"
        ),
        "video_patch_column_sector_order": {
            "front": [7, 8],
            "front_left": [5, 6],
            "left": [3, 4],
            "back_left": [1, 2],
            "back": [15, 0],
            "back_right": [13, 14],
            "right": [11, 12],
            "front_right": [9, 10],
        },
        "target_count": len(target_rows),
        "positive_target_count": sum(targets),
        "targets": target_rows,
        "training": False,
        "dev_test_or_evaluation_or_sealed_audit_labels_read": False,
        "note": (
            "This validates the annotation interval and deterministic coordinate "
            "mapping; it does not establish model quality or visual source detectability."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = verify_training_window()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
