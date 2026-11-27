"""Audit SoundActions v2 split and label metadata without loading media."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

RATER_FILES = tuple(f"Labels/SoundActions_labeling_{rater}.csv" for rater in "ABC")
MASTER_FILE = "Labels/SoundActions_labels_master.csv"
PERCEPTION_ALIASES = {
    "impulse": "impulsive",
    "impulsive": "impulsive",
    "sustain": "sustained",
    "sustained": "sustained",
    "iterative": "iterative",
    "unknown": "unknown",
    "": "missing",
    "can't tell": "unknown",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_zip_csv(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        return list(csv.DictReader(text))


def _record_number(value: str) -> str:
    match = re.search(r"(\d{1,3})", str(value))
    if not match:
        raise ValueError("record identifier does not contain a numeric ID")
    return str(int(match.group(1)))


def _normalized_perception(value: str) -> str:
    return PERCEPTION_ALIASES.get(str(value).strip().casefold(), "other")


def _label_vocabulary(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    labels = Counter(row[field].strip().casefold() for row in rows if row.get(field, "").strip())
    return {
        "distinct_values": len(labels),
        "singleton_values": sum(count == 1 for count in labels.values()),
        "observed_values": sum(labels.values()),
    }


def audit_metadata(overview_path: Path, label_archive_path: Path) -> dict[str, Any]:
    overview_rows = _read_csv(overview_path)
    required_overview = {
        "number",
        "duration_s",
        "perception_type",
        "wav",
        "mp4",
        "jpg",
    }
    if not overview_rows or not required_overview.issubset(overview_rows[0]):
        raise ValueError("overview CSV is empty or missing required fields")

    overview_ids = [_record_number(row["number"]) for row in overview_rows]
    if len(set(overview_ids)) != len(overview_ids):
        raise ValueError("overview has duplicate recording IDs")
    media_counts: dict[str, dict[str, int]] = {}
    for field, extension in (("wav", "wav"), ("mp4", "mp4"), ("jpg", "jpg")):
        names = [row[field].strip() for row in overview_rows]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"overview has missing or duplicate {field} names")
        ids = []
        for name in names:
            match = re.match(r"^SoundAction\s+(\d{3})\s+-\s+.+\." + extension + r"$", name)
            if not match:
                raise ValueError(f"overview contains a malformed {field} filename")
            ids.append(str(int(match.group(1))))
        if set(ids) != set(overview_ids):
            raise ValueError(f"overview {field} filenames do not map one-to-one to records")
        media_counts[field] = {"unique_references": len(set(names)), "missing": 0}

    with zipfile.ZipFile(label_archive_path) as archive:
        names = set(archive.namelist())
        needed = {*RATER_FILES, MASTER_FILE, "label-corrections.csv"}
        if not needed.issubset(names):
            raise ValueError("label archive is missing expected annotation files")
        raters = [_read_zip_csv(archive, name) for name in RATER_FILES]
        master = _read_zip_csv(archive, MASTER_FILE)
        corrections = _read_zip_csv(archive, "label-corrections.csv")

    if len(master) != len(overview_rows) or any(len(rows) != len(overview_rows) for rows in raters):
        raise ValueError("overview and annotation row counts differ")
    rater_ids = [{_record_number(row["SoundAction #"]) for row in rows} for rows in raters]
    master_ids = {_record_number(row["number"]) for row in master}
    if any(ids != set(overview_ids) for ids in rater_ids) or master_ids != set(overview_ids):
        raise ValueError("annotation recording IDs do not match overview IDs")

    rater_by_id = [{_record_number(row["SoundAction #"]): row for row in rows} for rows in raters]
    master_by_id = {_record_number(row["number"]): row for row in master}
    agreement: dict[str, dict[str, int]] = {}
    # Only PerceptionType has a stable common vocabulary across rater and master fields.
    perception_votes: Counter[str] = Counter()
    unanimous = 0
    majority = 0
    master_agree_matches = 0
    for record_id in overview_ids:
        values = [
            _normalized_perception(rater_by_id[i][record_id]["Schaffer mass type"])
            for i in range(3)
        ]
        counts = Counter(values)
        votes = max(counts.values())
        unanimous += votes == 3
        majority += votes >= 2
        perception_votes.update(values)
        declared = _normalized_perception(master_by_id[record_id]["perception_type"])
        winning = counts.most_common(1)[0][0] if votes >= 2 else "unknown"
        master_agree_matches += declared == winning
    agreement["perception_type"] = {
        "unanimous_items": unanimous,
        "majority_items": majority,
        "no_majority_items": len(overview_rows) - majority,
        "master_consensus_matches_majority": master_agree_matches,
        "rater_vote_counts": dict(sorted(perception_votes.items())),
    }

    correction_ids = {_record_number(row["number"]) for row in corrections}
    durations = [float(row["duration_s"]) for row in overview_rows]
    rater_label_rows = [row for rater_rows in raters for row in rater_rows]
    result = {
        "schema": "tiny-omni-decision/soundactions-metadata-audit-v1",
        "record_count": len(overview_rows),
        "unique_record_ids": len(set(overview_ids)),
        "annotation_rater_count": len(raters),
        "split_column_in_overview": any(
            field in overview_rows[0] for field in ("split", "partition", "fold")
        ),
        "media_references": media_counts,
        "duration_seconds": {
            "min": min(durations),
            "median": sorted(durations)[len(durations) // 2],
            "max": max(durations),
        },
        "perception_type_consensus_counts": dict(
            sorted(
                Counter(
                    _normalized_perception(row["perception_type"]) for row in overview_rows
                ).items()
            )
        ),
        "label_agreement": agreement,
        "free_text_label_vocabularies": {
            "action_rater_fields": _label_vocabulary(rater_label_rows, "Action"),
            "object_rater_fields": _label_vocabulary(rater_label_rows, "Objects"),
        },
        "published_correction_receipt": {
            "correction_cells": len(corrections),
            "affected_record_count": len(correction_ids),
        },
        "input_sha256": {
            "overview_csv": sha256_file(overview_path),
            "label_archive": sha256_file(label_archive_path),
        },
        "raw_titles_urls_and_ids_emitted": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overview", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = (
        json.dumps(audit_metadata(args.overview, args.labels), indent=2, sort_keys=True) + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
