"""Training-only label-density preflight for spatial Decision options.

Reads the pinned STARSS23 dev-train annotation members only. It does not read
media, dev-test labels, or the unlabeled evaluation release.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import platform
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from scripts.extract_frozen_videoprism_features import sha256_file

ROOT = Path("C:/CodexArtifacts/pretrained-reuse-r2")
SPLIT_PATH = ROOT / "starss23-room-splits-v1/train.jsonl"
METADATA_PATH = ROOT / "starss23-source-audit-01/metadata_dev.zip"
SPLIT_SHA256 = "571c2b537a809b2f92c9289fd932c20e0169ca1c0e38f4dc0f4c6ad6ef2ac978"
METADATA_SHA256 = "2fa848545a520b4a9d490459de7f079aa49ae16daf0affc859c016f5146981d6"
SECTORS = (
    "front",
    "front_left",
    "left",
    "back_left",
    "back",
    "back_right",
    "right",
    "front_right",
)
EVENT_NAMES = {
    0: "speech",
    2: "clapping",
    3: "telephone",
    4: "laughter",
    5: "domestic sounds",
    6: "footsteps",
    7: "door",
    8: "music",
    9: "instrument",
    10: "water tap",
    11: "bell",
    12: "knock",
}
EVENT_CLASSES = tuple(EVENT_NAMES)


def azimuth_sector(azimuth_degrees: int) -> int:
    """Map STARSS azimuth to one of eight 45-degree sectors; 0 degrees is front."""
    if isinstance(azimuth_degrees, bool) or not isinstance(azimuth_degrees, int):
        raise ValueError("azimuth must be an integer number of degrees")
    if not -180 <= azimuth_degrees <= 180:
        raise ValueError("azimuth must be in [-180, 180]")
    normalized = (azimuth_degrees + 180) % 360 - 180
    return math.floor((normalized + 22.5) / 45) % len(SECTORS)


def canonical_class(raw_class: int) -> int:
    """Merge female/male speech labels into the single proposed speech task."""
    if not 0 <= raw_class <= 12:
        raise ValueError("STARSS raw class must be in [0, 12]")
    return 0 if raw_class in (0, 1) else raw_class


def interval_sector_targets(rows, start_tick: int, stop_tick: int) -> list[int]:
    """Return 12x8 binary option targets for [start_tick, stop_tick)."""
    if start_tick < 0 or stop_tick <= start_tick:
        raise ValueError("annotation interval must be non-empty and non-negative")
    positives = set()
    for row in rows:
        if len(row) != 6:
            raise ValueError("expected six STARSS annotation fields")
        tick, raw_class, _source, azimuth, elevation, distance = map(int, row)
        if tick < 0 or not -180 <= azimuth <= 180 or not -90 <= elevation <= 90 or distance < 0:
            raise ValueError("invalid STARSS spatial annotation")
        if start_tick <= tick < stop_tick:
            positives.add((canonical_class(raw_class), azimuth_sector(azimuth)))
    return [
        int((event_class, sector) in positives)
        for event_class in EVENT_CLASSES
        for sector in range(len(SECTORS))
    ]


def summarize_training(split_path: Path, metadata_path: Path) -> dict:
    if sha256_file(split_path) != SPLIT_SHA256:
        raise ValueError("pinned STARSS23 training split changed")
    if sha256_file(metadata_path) != METADATA_SHA256:
        raise ValueError("pinned STARSS23 metadata archive changed")

    records = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 49 or any(row.get("split") != "train" for row in records):
        raise ValueError("expected the unchanged 49-recording training manifest")
    if len({row.get("clip") for row in records}) != len(records):
        raise ValueError("duplicate training recording IDs")

    grouped: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    rows_by_class = Counter()
    recordings_by_room = Counter()
    rows_by_room_class = defaultdict(Counter)
    clip_room = {}
    rooms = set()
    with zipfile.ZipFile(metadata_path) as archive:
        names = set(archive.namelist())
        for record in records:
            member = record.get("member", "")
            if not member.startswith("metadata_dev/dev-train-") or member not in names:
                raise ValueError("manifest referenced a non-training annotation member")
            room = record.get("room")
            if not room:
                raise ValueError("missing training room identity")
            rooms.add(room)
            recordings_by_room[room] += 1
            clip_room[record["clip"]] = room
            for row in csv.reader(io.StringIO(archive.read(member).decode("utf-8"))):
                if len(row) != 6:
                    raise ValueError("expected six STARSS annotation fields")
                tick, raw_class, source, azimuth, elevation, distance = map(int, row)
                if (
                    tick < 0
                    or source < 0
                    or not -180 <= azimuth <= 180
                    or not -90 <= elevation <= 90
                    or distance < 0
                ):
                    raise ValueError("invalid training spatial annotation")
                event_class = canonical_class(raw_class)
                sector = azimuth_sector(azimuth)
                grouped[(record["clip"], tick, event_class)].append(sector)
                rows_by_class[event_class] += 1
                rows_by_room_class[(room, event_class)]["annotation_rows"] += 1

    sector_occurrences = Counter()
    source_multiplicity = Counter()
    multi_sector = Counter()
    queries_by_room_class = defaultdict(Counter)
    for (clip, _, event_class), sectors in grouped.items():
        queries_by_room_class[(clip_room[clip], event_class)]["event_tick_queries"] += 1
        source_multiplicity[event_class] += len(sectors) > 1
        distinct = set(sectors)
        multi_sector[event_class] += len(distinct) > 1
        sector_occurrences.update({(event_class, sector): 1 for sector in distinct})

    return {
        "scope": "pinned 49-recording dev-train annotation members only",
        "source_hashes": {
            str(split_path): SPLIT_SHA256,
            str(metadata_path): METADATA_SHA256,
        },
        "script_sha256": sha256_file(Path(__file__)),
        "python": platform.python_version(),
        "recordings": len(records),
        "rooms": sorted(rooms),
        "room_count": len(rooms),
        "recordings_by_room": dict(sorted(recordings_by_room.items())),
        "annotation_rows_after_speech_merge": dict(sorted(rows_by_class.items())),
        "annotation_rows_by_room_and_class": {
            f"{room}:{event_class}": dict(sorted(counts.items()))
            for (room, event_class), counts in sorted(rows_by_room_class.items())
        },
        "event_tick_queries_by_room_and_class": {
            f"{room}:{event_class}": dict(sorted(counts.items()))
            for (room, event_class), counts in sorted(queries_by_room_class.items())
        },
        "unique_recording_tick_event_queries": len(grouped),
        "queries_with_multiple_source_rows": dict(sorted(source_multiplicity.items())),
        "queries_with_multiple_direction_sectors": dict(sorted(multi_sector.items())),
        "unique_recording_tick_event_sector_positives": {
            f"{event_class}:{sector}": count
            for (event_class, sector), count in sorted(sector_occurrences.items())
        },
        "sector_options": list(SECTORS),
        "media_read": False,
        "dev_test_annotations_read": False,
        "official_evaluation_read": False,
        "calibration_validation_read": False,
        "sealed_audit_read": False,
        "training": False,
        "note": (
            "Counts are grouped by recording/tick/event; these are correlated labels, "
            "not independent examples or model quality metrics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = summarize_training(SPLIT_PATH, METADATA_PATH)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
