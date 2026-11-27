"""Audit BSD10k metadata, per-file rights, labels, and uploader-group splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ALLOWED_LICENSE_PATHS = {
    "publicdomain/zero/1.0": "CC0-1.0",
    "licenses/by/4.0": "CC-BY-4.0",
}
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REQUIRED_FIELDS = {
    "sound_id",
    "class",
    "confidence",
    "uploader",
    "license",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_license(value: str) -> str:
    path = "/".join(part for part in urlparse(value.strip()).path.casefold().split("/") if part)
    return ALLOWED_LICENSE_PATHS.get(path, f"OTHER:{path or 'missing'}")


def canonical_license_with_explicit_extras(value: str, extra_allowed: set[str]) -> str:
    """Recognize an explicitly opted-in license without changing the default allowlist."""
    path = "/".join(part for part in urlparse(value.strip()).path.casefold().split("/") if part)
    known_extras = {"licenses/by/3.0": "CC-BY-3.0"}
    name = known_extras.get(path)
    if name is not None and name in extra_allowed:
        return name
    return canonical_license(value)


def _group_split(rows: list[dict[str, str]], seed: int = 17) -> dict[str, str]:
    """Greedily balance class counts while keeping every uploader in one split."""
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group_counts[row["uploader"].strip().casefold()][row["class"].strip()] += 1

    totals = Counter()
    groups_per_class = Counter()
    for counts in group_counts.values():
        totals.update(counts)
        groups_per_class.update(counts.keys())
    total_groups = len(group_counts)
    target_groups = {
        "train": int(total_groups * 0.70),
        "validation": int(total_groups * 0.15),
    }
    target_groups["test"] = total_groups - sum(target_groups.values())
    assigned_counts = {name: Counter() for name in SPLIT_FRACTIONS}
    assigned_class_groups = {name: Counter() for name in SPLIT_FRACTIONS}
    assigned_rows = Counter()
    assigned_groups = Counter()
    assignments: dict[str, str] = {}

    def tie_key(group: str) -> str:
        return hashlib.sha256(f"bsd10k-v1.2:{seed}:{group}".encode()).hexdigest()

    def rarity(group: str) -> float:
        counts = group_counts[group]
        return max(1.0 / groups_per_class[label] for label in counts)

    group_order = sorted(
        group_counts,
        key=lambda group: (-rarity(group), -sum(group_counts[group].values()), tie_key(group)),
    )
    for group in group_order:
        counts = group_counts[group]
        choices: list[tuple[float, str]] = []
        for split, fraction in SPLIT_FRACTIONS.items():
            if assigned_groups[split] >= target_groups[split]:
                continue
            class_cost = 0.0
            for label, total in totals.items():
                target = max(total * fraction, 1.0)
                before = assigned_counts[split][label] - target
                after = assigned_counts[split][label] + counts[label] - target
                class_cost += (after * after - before * before) / (target * target)
                target_class_groups = max(groups_per_class[label] * fraction, 1.0)
                before_groups = assigned_class_groups[split][label] - target_class_groups
                after_groups = (
                    assigned_class_groups[split][label] + int(label in counts) - target_class_groups
                )
                class_cost += (after_groups * after_groups - before_groups * before_groups) / (
                    target_class_groups * target_class_groups
                )
            target_rows = max(sum(totals.values()) * fraction, 1.0)
            before_rows = assigned_rows[split] - target_rows
            after_rows = assigned_rows[split] + sum(counts.values()) - target_rows
            class_cost += len(totals) * (
                (after_rows * after_rows - before_rows * before_rows) / (target_rows * target_rows)
            )
            choices.append((class_cost, split))
        _, selected = min(choices, key=lambda item: (item[0], item[1]))
        assignments[group] = selected
        assigned_counts[selected].update(counts)
        assigned_class_groups[selected].update(counts.keys())
        assigned_rows[selected] += sum(counts.values())
        assigned_groups[selected] += 1
    return assignments


def audit_metadata(archive_path: Path, seed: int = 17) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            name for name in archive.namelist() if name.casefold().endswith("bsd10k_metadata.csv")
        ]
        if len(candidates) != 1:
            raise ValueError("archive must contain exactly one BSD10k_metadata.csv")
        with archive.open(candidates[0]) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
                raise ValueError("metadata CSV is missing required fields")
            rows = list(reader)

    if not rows:
        raise ValueError("metadata CSV contains no records")
    ids = [row["sound_id"].strip() for row in rows]
    if any(not value for value in ids):
        raise ValueError("metadata contains a missing sound_id")
    if len(set(ids)) != len(ids):
        raise ValueError("metadata contains duplicate sound_id values")
    for row in rows:
        if not row["class"].strip() or not row["uploader"].strip():
            raise ValueError("metadata contains a missing class or uploader group")
        try:
            confidence = int(row["confidence"])
        except ValueError as exc:
            raise ValueError("metadata contains an invalid confidence value") from exc
        if confidence not in range(1, 6):
            raise ValueError("confidence must be an integer from 1 through 5")

    license_counts = Counter(canonical_license(row["license"]) for row in rows)
    eligible = [
        row for row in rows if canonical_license(row["license"]) in ALLOWED_LICENSE_PATHS.values()
    ]
    if not eligible:
        raise ValueError("no records meet the exact per-file license allowlist")

    split_by_group = _group_split(eligible, seed)
    classes = sorted({row["class"].strip() for row in eligible})
    class_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    group_sizes: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_FRACTIONS}
    split_groups: dict[str, set[str]] = {name: set() for name in SPLIT_FRACTIONS}
    for row in eligible:
        label = row["class"].strip()
        group = row["uploader"].strip().casefold()
        split = split_by_group[group]
        class_records[label].append(row)
        group_sizes[group] += 1
        confidence_counts[row["confidence"].strip()] += 1
        split_counts[split][label] += 1
        split_groups[split].add(group)

    class_support: dict[str, dict[str, Any]] = {}
    for label in classes:
        label_rows = class_records[label]
        label_groups = {row["uploader"].strip().casefold() for row in label_rows}
        class_support[label] = {
            "examples": len(label_rows),
            "unique_uploader_groups": len(label_groups),
            "confidence_counts": dict(
                sorted(Counter(row["confidence"] for row in label_rows).items())
            ),
            "split_examples": {split: split_counts[split][label] for split in SPLIT_FRACTIONS},
            "split_uploader_groups": {
                split: sum(1 for group in label_groups if split_by_group[group] == split)
                for split in SPLIT_FRACTIONS
            },
        }

    split_example_counts = {split: sum(split_counts[split].values()) for split in SPLIT_FRACTIONS}
    split_group_counts = {split: len(groups) for split, groups in split_groups.items()}
    overlap_count = sum(
        len(split_groups[left] & split_groups[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    return {
        "schema": "tiny-omni-decision/bsd10k-metadata-preflight-v1",
        "record_count": len(rows),
        "unique_sound_ids": len(set(ids)),
        "license_counts": dict(sorted(license_counts.items())),
        "eligible_license_families": ["CC0-1.0", "CC-BY-4.0"],
        "eligible_records": len(eligible),
        "eligible_unique_uploader_groups": len(group_sizes),
        "largest_uploader_group_examples": max(group_sizes.values()),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "classes": class_support,
        "group_split_preview": {
            "algorithm": (
                "deterministic greedy class/example/group balance, indivisible uploader groups"
            ),
            "seed": seed,
            "fractions_are_group_targets": SPLIT_FRACTIONS,
            "examples": split_example_counts,
            "uploader_groups": split_group_counts,
            "uploader_overlap_count": overlap_count,
        },
        "raw_identifiers_titles_and_urls_emitted": False,
        "media_payload_or_hashes_inspected": False,
    }


def build_split_manifest(archive_path: Path, seed: int = 17) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            name for name in archive.namelist() if name.casefold().endswith("bsd10k_metadata.csv")
        ]
        if len(candidates) != 1:
            raise ValueError("archive must contain exactly one BSD10k_metadata.csv")
        with archive.open(candidates[0]) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
                raise ValueError("metadata CSV is missing required fields")
            rows = list(reader)
    eligible = [
        row for row in rows if canonical_license(row["license"]) in ALLOWED_LICENSE_PATHS.values()
    ]
    if len({row["sound_id"].strip() for row in eligible}) != len(eligible):
        raise ValueError("allowlisted metadata contains duplicate sound IDs")
    assignments = _group_split(eligible, seed)
    examples = []
    for row in sorted(eligible, key=lambda item: int(item["sound_id"])):
        uploader = row["uploader"].strip().casefold()
        examples.append(
            {
                "sound_id": row["sound_id"].strip(),
                "class": row["class"].strip(),
                "license": canonical_license(row["license"]),
                "uploader_group_sha256": hashlib.sha256(uploader.encode()).hexdigest(),
                "split": assignments[uploader],
            }
        )
    return {
        "schema": "tiny-omni-decision/bsd10k-group-split-v1",
        "source_record_id": "19868804",
        "metadata_zip_sha256": sha256_file(archive_path),
        "seed": seed,
        "group_key": "normalized Freesound uploader",
        "algorithm": (
            "deterministic greedy class/example/group balance, indivisible uploader groups"
        ),
        "fractions_are_group_targets": SPLIT_FRACTIONS,
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-manifest-output", type=Path)
    args = parser.parse_args()
    result = audit_metadata(args.metadata_zip, args.seed)
    result["metadata_zip_sha256"] = sha256_file(args.metadata_zip)
    if args.split_manifest_output:
        split_manifest = build_split_manifest(args.metadata_zip, args.seed)
        args.split_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        split_rendered = json.dumps(split_manifest, indent=2, sort_keys=True) + "\n"
        args.split_manifest_output.write_text(split_rendered, encoding="utf-8")
        result["split_manifest_sha256"] = hashlib.sha256(split_rendered.encode("utf-8")).hexdigest()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
