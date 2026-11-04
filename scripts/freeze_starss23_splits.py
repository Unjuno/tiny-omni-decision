"""Freeze STARSS23 room splits and inspect training annotations only."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from scripts.fetch_starss23_smoke import PINNED, metadata_inventory
from scripts.probe_starss23_windows import EVENTS, parse_annotations

SEED = 17
EXPOSED_ROOMS = frozenset({"tau:room12", "tau:room13"})


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def split_records(records, missing_video):
    rows = [dict(r) for r in records if r["split"] == "train" and r["clip"] not in missing_video]
    rooms = {r["room"] for r in rows}
    if not EXPOSED_ROOMS <= rooms:
        raise ValueError("exposed training rooms missing")
    unseen = sorted(rooms - EXPOSED_ROOMS, key=lambda r: stable_hash([SEED, r]))
    if len(unseen) < 4:
        raise ValueError("insufficient unexposed rooms")
    assigned = {unseen[0]: "calibration", unseen[1]: "validation", unseen[2]: "validation"}
    for row in rows:
        row["upstream_split"] = row["split"]
        row["split"] = assigned.get(row["room"], "train")
        row["source_revision"] = "doi:10.5281/zenodo.7880637"
    return sorted(rows, key=lambda r: r["clip"])


def train_coverage(archive, rows):
    counts = Counter()
    clips = Counter()
    for row in rows:
        if row["split"] != "train":
            continue
        annotations = parse_annotations(archive.read(row["member"]).decode("utf-8"))
        active = {(frame, label) for frame, label, _ in annotations}
        counts.update(label for _, label in active)
        clips.update({label for _, label in active})
    return [
        {"event": event, "active_ticks": counts[i], "recordings": clips[i]}
        for i, event in enumerate(EVENTS)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata_hash = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    if metadata_hash != PINNED["metadata_dev.zip"]:
        raise ValueError("source metadata hash mismatch")
    probe = json.loads(args.source_probe.read_text(encoding="utf-8"))
    if probe["record"] != "doi:10.5281/zenodo.7880637" or probe["status"] != "PASS":
        raise ValueError("invalid source probe")
    with zipfile.ZipFile(args.metadata) as archive:
        records = metadata_inventory(archive.namelist())
        rows = split_records(records, set(probe["missing_video_clips"]))
        coverage = train_coverage(archive, rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "source_revision": probe["record"],
        "metadata_sha256": metadata_hash,
        "source_probe_sha256": hashlib.sha256(args.source_probe.read_bytes()).hexdigest(),
        "seed": SEED,
        "exposed_rooms_forced_train": sorted(EXPOSED_ROOMS),
        "assignment": "SHA256 canonical JSON [17,room]; first unseen cal, next two val",
        "splits": {},
        "training_annotation_coverage": coverage,
        "calibration_validation_labels_read": False,
        "dev_test_contents_read": False,
    }
    for split in ("train", "calibration", "validation"):
        selected = [r for r in rows if r["split"] == split]
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in selected).encode()
        (args.output_dir / f"{split}.jsonl").write_bytes(payload)
        summary["splits"][split] = {
            "recordings": len(selected),
            "rooms": sorted({r["room"] for r in selected}),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    (args.output_dir / "result.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
