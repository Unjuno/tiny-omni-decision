"""Read-only provenance and split audit for a local MAVD-traffic source bundle.

The audit deliberately reads only TRAIN and validation annotation payloads.
For TEST it hashes the archive and lists member names from the ZIP directory,
but never opens or decodes a test annotation member.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

ZENODO_RECORD = "https://zenodo.org/records/4741232"
EXPECTED_MD5 = {
    "annotations_train.zip": "ea67a330ab695873a12a07ed494eae42",
    "annotations_validate.zip": "30edae5c398dffe52a2ee1373e89fd96",
    "annotations_test.zip": "94e609cad8702f5fb6afe52815547b67",
    "metadata.txt": "593cad71171d713775e102bb59d909da",
    "README": "f71d547c558d007593e8ec5b70aa696c",
}
FILE_PATTERN = re.compile(r"^(?P<site>[a-z]+)\d{4}_\d{2}$")


def file_hashes(path: Path) -> dict[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def _member_stems(archive: Path) -> list[str]:
    with ZipFile(archive) as zipped:
        return sorted(
            Path(name).stem for name in zipped.namelist() if name.lower().endswith(".txt")
        )


def _read_development_labels(archive: Path) -> tuple[Counter[str], int, list[str]]:
    labels: Counter[str] = Counter()
    total = 0
    invalid: list[str] = []
    with ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if not name.lower().endswith(".txt"):
                continue
            for line_number, line in enumerate(
                zipped.read(name).decode("utf-8-sig", errors="strict").splitlines(), 1
            ):
                fields = line.split("\t")
                if len(fields) != 3:
                    invalid.append(f"{name}:{line_number}:field_count")
                    continue
                try:
                    start, end = float(fields[0]), float(fields[1])
                except ValueError:
                    invalid.append(f"{name}:{line_number}:non_numeric_time")
                    continue
                if not (0 <= start < end):
                    invalid.append(f"{name}:{line_number}:invalid_interval")
                    continue
                labels[fields[2]] += 1
                total += 1
    return labels, total, invalid


def _load_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stem = Path(row["file"]).stem
            match = FILE_PATTERN.fullmatch(stem)
            if match is None or stem in rows:
                raise ValueError(f"invalid or duplicate metadata file ID: {stem}")
            rows[stem] = {**row, "site": match.group("site")}
    return rows


def audit_source(source_dir: Path) -> dict[str, object]:
    source_dir = source_dir.resolve(strict=True)
    hashes: dict[str, dict[str, str]] = {}
    for filename, expected_md5 in EXPECTED_MD5.items():
        path = source_dir / filename
        actual = file_hashes(path)
        if actual["md5"] != expected_md5:
            raise ValueError(f"source MD5 mismatch for {filename}: {actual['md5']}")
        hashes[filename] = actual

    metadata = _load_metadata(source_dir / "metadata.txt")
    members = {
        split: _member_stems(source_dir / f"annotations_{split}.zip")
        for split in ("train", "validate", "test")
    }
    for split, split_members in members.items():
        missing = sorted(set(split_members) - set(metadata))
        if missing:
            raise ValueError(f"{split} annotation IDs missing metadata: {missing}")

    groups: dict[str, str] = {}
    for stem, row in metadata.items():
        groups[stem] = f"{row['site']}:{row['year']}-{row['month']}-{row['day']}"
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for split, split_members in members.items():
        for stem in split_members:
            splits_by_group[groups[stem]].add(split)

    group_overlap = {
        group: sorted(splits)
        for group, splits in sorted(splits_by_group.items())
        if len(splits) > 1
    }
    development_labels: dict[str, dict[str, int]] = {}
    development_counts: dict[str, int] = {}
    invalid_intervals: dict[str, list[str]] = {}
    for split in ("train", "validate"):
        labels, count, invalid = _read_development_labels(source_dir / f"annotations_{split}.zip")
        development_labels[split] = dict(sorted(labels.items()))
        development_counts[split] = count
        invalid_intervals[split] = invalid

    return {
        "dataset": "MAVD-traffic",
        "zenodo_record": ZENODO_RECORD,
        "zenodo_record_id": 4741232,
        "record_version": "0.1.1",
        "license": "CC-BY-4.0",
        "source_dir": str(source_dir),
        "source_file_hashes": hashes,
        "metadata_record_count": len(metadata),
        "annotation_file_counts": {split: len(ids) for split, ids in members.items()},
        "sites_by_split": {
            split: dict(sorted(Counter(metadata[stem]["site"] for stem in ids).items()))
            for split, ids in members.items()
        },
        "site_day_groups_by_split": {
            split: sorted({groups[stem] for stem in ids}) for split, ids in members.items()
        },
        "cross_split_site_day_groups": group_overlap,
        "development_label_interval_counts": development_counts,
        "development_labels_by_split": development_labels,
        "development_invalid_intervals": invalid_intervals,
        "test_member_names_only": members["test"],
        "test_annotation_payloads_read": False,
        "media_payloads_downloaded_or_opened": False,
        "interpretation": (
            "Official train and validation annotations share site-day groups; "
            "they are not independent under this conservative capture grouping. "
            "The official test member names are recorded for grouping only; "
            "test annotation contents were not read."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_source(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "annotation_file_counts": result["annotation_file_counts"],
                "cross_split_site_day_groups": result["cross_split_site_day_groups"],
                "development_label_interval_counts": result["development_label_interval_counts"],
                "test_annotation_payloads_read": result["test_annotation_payloads_read"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
