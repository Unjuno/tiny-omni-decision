"""Build room-grouped binary event/azimuth Decision rows from dev-train labels."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.extract_starss23_corpus_features import read_manifest
from scripts.fetch_starss23_frozen_media import write_json
from scripts.preflight_starss23_spatial_decision import (
    EVENT_CLASSES,
    EVENT_NAMES,
    METADATA_PATH,
    METADATA_SHA256,
    SECTORS,
    interval_sector_targets,
)

WINDOWS_DIR = Path("C:/CodexArtifacts/pretrained-reuse-r2/starss23-all-windows-v1")
SPLITS_DIR = Path("C:/CodexArtifacts/pretrained-reuse-r2/starss23-room-splits-v1")
WINDOWS_RESULT_SHA256 = "8f08d3467bc49677fa432733b1bd3217b90056b649ad5c898202b6c97497c1a0"
SPLIT_RESULT_SHA256 = "209423b79c3bf1b98ce926aabf92a7f1f8262a5a8dea69e619f893fe9e1357b6"
SPLIT_FILE_SHAS = {
    "train": "571c2b537a809b2f92c9289fd932c20e0169ca1c0e38f4dc0f4c6ad6ef2ac978",
    "calibration": "0c0961a7c4eb33dddec6fe904b4fca26881a8e015a885ce2f6d3594c182f440f",
    "validation": "e1dd3789abd2c909fc66bf67aea26e8d1afa044458818a262470e54848d90b50",
}


def load_room_splits() -> tuple[dict[str, dict], dict[str, str]]:
    if sha256_file(WINDOWS_DIR / "result.json") != WINDOWS_RESULT_SHA256:
        raise ValueError("pinned fixed-window manifest changed")
    if sha256_file(SPLITS_DIR / "result.json") != SPLIT_RESULT_SHA256:
        raise ValueError("pinned room-split result changed")
    if sha256_file(METADATA_PATH) != METADATA_SHA256:
        raise ValueError("pinned STARSS23 metadata archive changed")
    records = {}
    split_by_recording = {}
    for split, expected_hash in SPLIT_FILE_SHAS.items():
        path = SPLITS_DIR / f"{split}.jsonl"
        if sha256_file(path) != expected_hash:
            raise ValueError(f"pinned {split} recording split changed")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            clip = row.get("clip")
            if not clip or row.get("split") != split or clip in records:
                raise ValueError("duplicate or misrouted recording in frozen room splits")
            if row.get("source_revision") != "doi:10.5281/zenodo.7880637":
                raise ValueError("unexpected STARSS23 source revision")
            if not row.get("member", "").startswith("metadata_dev/dev-train-"):
                raise ValueError("room split references a non-dev-train annotation")
            records[clip] = row
            split_by_recording[clip] = split
    expected_recordings = {"train": 49, "calibration": 8, "validation": 21}
    actual_recordings = Counter(split_by_recording.values())
    if actual_recordings != Counter(expected_recordings):
        raise ValueError("frozen room split recording counts changed")
    rooms_by_split = {
        split: {row["room"] for row in records.values() if row["split"] == split}
        for split in expected_recordings
    }
    if any(
        rooms_by_split[left] & rooms_by_split[right]
        for index, left in enumerate(expected_recordings)
        for right in list(expected_recordings)[index + 1 :]
    ):
        raise ValueError("room identity overlaps across task splits")
    return records, split_by_recording


def build_window_targets(window, annotations) -> list[dict]:
    if window["end_seconds"] - window["start_seconds"] != 5:
        raise ValueError("expected fixed five-second intervals")
    start_tick = round(window["start_seconds"] * 10)
    stop_tick = round(window["end_seconds"] * 10)
    labels = interval_sector_targets(annotations, start_tick, stop_tick)
    return [
        {
            "sample_id": f"{window['id']}:event-{event_class}:sector-{SECTORS[sector_index]}",
            "window_id": window["id"],
            "window_index": window["window_index"],
            "recording": window["recording"],
            "room": window["room"],
            "split": window["split"],
            "event_class": event_class,
            "event": EVENT_NAMES[event_class],
            "sector": SECTORS[sector_index],
            "question": (
                f"Is {EVENT_NAMES[event_class]} present in the "
                f"{SECTORS[sector_index].replace('_', ' ')} sector during this interval?"
            ),
            "options": ["no", "yes"],
            "target": labels[event_index * len(SECTORS) + sector_index],
        }
        for event_index, event_class in enumerate(EVENT_CLASSES)
        for sector_index in range(len(SECTORS))
    ]


def build_corpus(output_dir: Path) -> dict:
    records, split_by_recording = load_room_splits()
    windows = read_manifest(WINDOWS_DIR)
    indexed_windows = []
    media_splits = {"audio": {}, "video": {}}
    for index, row in enumerate(windows):
        clip = row["recording"]
        if clip not in records or split_by_recording[clip] != row["split"]:
            raise ValueError("window and room split identities disagree")
        if row["room"] != records[clip]["room"]:
            raise ValueError("window room differs from frozen split manifest")
        for modality in media_splits:
            media_sha = row["media"][modality]["sha256"]
            previous_split = media_splits[modality].setdefault(media_sha, row["split"])
            if previous_split != row["split"]:
                raise ValueError(f"{modality} media identity overlaps task splits")
        indexed_windows.append({**row, "window_index": index})
    if {row["recording"] for row in windows} != set(records):
        raise ValueError("fixed-window manifest and 78-recording room split differ")

    split_rows = {split: [] for split in SPLIT_FILE_SHAS}
    targets_by_split = {split: Counter() for split in SPLIT_FILE_SHAS}
    with zipfile.ZipFile(METADATA_PATH) as archive:
        archive_names = set(archive.namelist())
        annotation_cache = {}
        for clip, record in records.items():
            member = record["member"]
            if member not in archive_names:
                raise ValueError(f"missing pinned dev-train annotation: {member}")
            annotation_cache[clip] = list(
                csv.reader(io.StringIO(archive.read(member).decode("utf-8")))
            )
    for window in indexed_windows:
        rows = build_window_targets(window, annotation_cache[window["recording"]])
        split_rows[window["split"]].extend(rows)
        targets_by_split[window["split"]].update(
            (row["event"], row["sector"]) for row in rows if row["target"]
        )

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "RUNNING",
        "source_revision": "doi:10.5281/zenodo.7880637",
        "metadata_sha256": METADATA_SHA256,
        "windows_manifest_sha256": WINDOWS_RESULT_SHA256,
        "room_split_result_sha256": SPLIT_RESULT_SHA256,
        "room_split_file_sha256": SPLIT_FILE_SHAS,
        "split_grouping": "room first, recording inherited; fixed 5-second windows remain together",
        "target_definition": (
            "1 if any same-class annotation in the 100-ms half-open interval "
            "maps to the requested 45-degree azimuth sector"
        ),
        "speech_label_mapping": (
            "raw classes 0 and 1 merge into class 0; remaining source class ids "
            "are preserved"
        ),
        "event_classes": [
            {"id": event_class, "name": EVENT_NAMES[event_class]} for event_class in EVENT_CLASSES
        ],
        "sectors": list(SECTORS),
        "options": ["no", "yes"],
        "windows": len(windows),
        "recordings": len(records),
        "annotation_members_read": len(annotation_cache),
        "official_dev_test_or_evaluation_or_sealed_audit_labels_read": False,
        "training": False,
        "splits": {},
    }
    for split, rows in split_rows.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        result["splits"][split] = {
            "questions": len(rows),
            "windows": sum(1 for window in indexed_windows if window["split"] == split),
            "positive_questions": sum(int(row["target"]) for row in rows),
            "positives_by_event_sector": {
                f"{event}|{sector}": count
                for (event, sector), count in sorted(targets_by_split[split].items())
            },
            "sha256": sha256_file(path),
        }
    result["status"] = "PASS"
    write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_corpus(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
