"""Metadata-only split audit helpers for AVMIT feature experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLIT_BUCKETS = (
    ("train", 0, 69),
    ("calibration", 70, 79),
    ("validation", 80, 89),
    ("sealed_audit", 90, 99),
)


def normalize_source_id(value: str) -> str:
    """Normalize a publisher-relative path without collapsing distinct names."""
    return unicodedata.normalize("NFC", value.strip().replace("\\", "/")).casefold()


def deterministic_split(source_id: str, seed: int = 17) -> str:
    normalized = normalize_source_id(source_id)
    digest = hashlib.sha256(f"avmit-split-v1\0{seed}\0{normalized}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    for split, lower, upper in SPLIT_BUCKETS:
        if lower <= bucket <= upper:
            return split
    raise AssertionError(f"unassigned split bucket: {bucket}")


def _read_test_ids(path: Path) -> tuple[set[str], set[str], int]:
    ids: set[str] = set()
    source_assets: set[str] = set()
    row_count = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            # Only identity columns are used; published test targets are ignored.
            for column in ("filename", "new_filename", "tfrecord_filename"):
                value = row.get(column, "").strip()
                if value:
                    ids.add(normalize_source_id(value))
                    if column == "filename":
                        source_assets.add(normalize_source_id(value))
    return ids, source_assets, row_count


def audit_avmit_metadata(
    ratings_csv: Path,
    test_set_csv: Path,
    output_dir: Path,
    *,
    seed: int = 17,
    source_revision: str = "e3c06ec0e215b1934d141f86008e92e567366d6c",
) -> dict[str, Any]:
    """Freeze clip-group splits while never serializing sealed targets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    test_ids, test_source_assets, test_row_count = _read_test_ids(test_set_csv)
    candidates: dict[str, dict[str, str]] = {}
    location_counts: Counter[str] = Counter()
    excluded_test = 0
    target_counts: dict[str, Counter[str]] = {
        name: Counter() for name, _, _ in SPLIT_BUCKETS if name != "sealed_audit"
    }
    rater_support_counts: dict[str, Counter[str]] = {
        name: Counter() for name, _, _ in SPLIT_BUCKETS if name != "sealed_audit"
    }
    vote_target_counts: dict[str, Counter[str]] = {
        name: Counter() for name, _, _ in SPLIT_BUCKETS if name != "sealed_audit"
    }

    with ratings_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "filename",
            "tfrecord_filename",
            "video_location",
            "AVMIT_label",
            "r1",
            "r2",
            "r3",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"ratings CSV missing columns: {sorted(required - set(reader.fieldnames or []))}"
            )
        for row in reader:
            location = row["video_location"].strip().casefold()
            location_counts[location] += 1
            # Only publisher TRAIN records are eligible. Published source validation
            # and the exposed AVMIT test set are excluded before target access.
            if location != "training":
                continue
            source_id = normalize_source_id(row["filename"])
            feature_record = row["tfrecord_filename"].strip().replace("\\", "/")
            normalized_feature_record = normalize_source_id(feature_record)
            if source_id in test_ids or normalized_feature_record in test_ids:
                excluded_test += 1
                continue
            if source_id in candidates:
                raise ValueError(f"duplicate source video identity: {source_id}")
            split = deterministic_split(source_id, seed)
            item = {
                "asset_id_sha256": hashlib.sha256(source_id.encode()).hexdigest(),
                "feature_record": feature_record,
            }
            if split != "sealed_audit":
                label = row["AVMIT_label"].strip()
                if not label:
                    raise ValueError(f"missing training/development label for {source_id}")
                item["target_label"] = label
                target_counts[split][label] += 1
                votes = tuple(int(row[name]) for name in ("r1", "r2", "r3"))
                if sum(votes) != 3:
                    raise ValueError(f"expected three annotator votes for {source_id}")
                for support in votes:
                    rater_support_counts[split][str(support)] += 1
                if max(votes) >= 2:
                    vote_target_counts[split][str(votes.index(max(votes)) + 1)] += 1
                else:
                    vote_target_counts[split]["all_three_disagree"] += 1
            candidates[source_id] = {**item, "split": split}

    feature_names = [normalize_source_id(item["feature_record"]) for item in candidates.values()]
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("feature record maps to more than one source video")

    output_hashes: dict[str, dict[str, Any]] = {}
    all_asset_ids: dict[str, set[str]] = {}
    all_feature_ids: dict[str, set[str]] = {}
    for split, _, _ in SPLIT_BUCKETS:
        rows = sorted(
            (item for item in candidates.values() if item["split"] == split),
            key=lambda item: item["asset_id_sha256"],
        )
        payload = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        )
        split_path = output_dir / f"{split}.jsonl"
        split_path.write_text(payload, encoding="utf-8")
        asset_ids = {row["asset_id_sha256"] for row in rows}
        feature_ids = {normalize_source_id(row["feature_record"]) for row in rows}
        all_asset_ids[split] = asset_ids
        all_feature_ids[split] = feature_ids
        output_hashes[split] = {
            "rows": len(rows),
            "unique_asset_groups": len(asset_ids),
            "unique_feature_records": len(feature_ids),
            "id_order_sha256": hashlib.sha256(
                "\n".join(row["asset_id_sha256"] for row in rows).encode()
            ).hexdigest(),
            "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "action_class_counts": dict(sorted(target_counts[split].items()))
            if split != "sealed_audit"
            else None,
            "rater_vote_target_counts": dict(sorted(vote_target_counts[split].items()))
            if split != "sealed_audit"
            else None,
            "rater_support_histogram": dict(sorted(rater_support_counts[split].items()))
            if split != "sealed_audit"
            else None,
        }
    split_names = [name for name, _, _ in SPLIT_BUCKETS]
    asset_overlap_count = 0
    feature_overlap_count = 0
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            asset_overlap_count += len(all_asset_ids[left] & all_asset_ids[right])
            feature_overlap_count += len(all_feature_ids[left] & all_feature_ids[right])
    summary: dict[str, Any] = {
        "source": "Audiovisual Moments in Time (AVMIT)",
        "source_revision": source_revision,
        "ratings_csv_sha256": hashlib.sha256(ratings_csv.read_bytes()).hexdigest(),
        "test_set_csv_sha256": hashlib.sha256(test_set_csv.read_bytes()).hexdigest(),
        "split_seed": seed,
        "split_rule": (
            "SHA256('avmit-split-v1\\0seed\\0normalized publisher filename') mod 100; "
            "train 0-69, calibration 70-79, validation 80-89, sealed_audit 90-99"
        ),
        "ratings_rows_by_publisher_location": dict(sorted(location_counts.items())),
        "published_test_rows": test_row_count,
        "published_test_source_asset_ids": len(test_source_assets),
        "published_test_identity_tokens": len(test_ids),
        "training_rows_excluded_by_published_test_identity": excluded_test,
        "eligible_source_videos": len(candidates),
        "cross_split_source_identity_overlap": asset_overlap_count,
        "cross_split_feature_record_overlap": feature_overlap_count,
        "splits": output_hashes,
        "limitations": [
            (
                "Only source filename and feature-record identity were checked; no raw media or "
                "normalized-content hashes were available."
            ),
            (
                "The sealed audit manifest contains opaque IDs and feature-record paths only; "
                "sealed labels were not serialized or counted."
            ),
            (
                "The publisher's publicly released 960-video test set was excluded by identity; "
                "no test labels were used."
            ),
            (
                "Publisher training source groups are clip identities; parent-upload identity "
                "beyond the released clip filename is unknown."
            ),
        ],
    }
    summary_path = output_dir / "metadata-audit.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    return summary


def build_savc_pair_rows(
    split_rows: list[dict[str, str]], split_name: str
) -> tuple[list[dict[str, str]], list[str]]:
    """Build balanced, disjoint SAVC pair units from one labeled split."""
    if split_name not in {"train", "calibration", "validation"}:
        raise ValueError("SAVC pairing only accepts train, calibration, or validation")
    if any("target_label" not in row for row in split_rows):
        raise ValueError("SAVC pairing requires labels from a non-sealed split")

    classes: dict[str, list[dict[str, str]]] = defaultdict(list)
    asset_ids: set[str] = set()
    feature_records: set[str] = set()
    for row in split_rows:
        asset_id = row["asset_id_sha256"]
        feature_record = normalize_source_id(row["feature_record"])
        if asset_id in asset_ids:
            raise ValueError(f"duplicate source clip in split: {asset_id}")
        if feature_record in feature_records:
            raise ValueError(f"duplicate feature record in split: {feature_record}")
        asset_ids.add(asset_id)
        feature_records.add(feature_record)
        classes[row["target_label"]].append(row)

    for rows in classes.values():
        rows.sort(key=lambda row: row["asset_id_sha256"])

    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    while True:
        available = sorted(
            ((len(rows), label) for label, rows in classes.items() if rows),
            key=lambda entry: (-entry[0], entry[1]),
        )
        if len(available) < 2:
            break
        left_label = available[0][1]
        right_label = available[1][1]
        pairs.append((classes[left_label].pop(0), classes[right_label].pop(0)))

    leftovers = sorted(row["asset_id_sha256"] for rows in classes.values() for row in rows)
    output: list[dict[str, str]] = []
    for left, right in pairs:
        group_material = "\0".join(sorted((left["asset_id_sha256"], right["asset_id_sha256"])))
        pair_group = hashlib.sha256(
            f"avmit-savc-pair-v1\0{split_name}\0{group_material}".encode()
        ).hexdigest()
        entries = (
            (left, left, "same_action"),
            (right, right, "same_action"),
            (left, right, "different_action"),
            (right, left, "different_action"),
        )
        for audio_source, video_source, target in entries:
            output.append(
                {
                    "split": split_name,
                    "pair_group_sha256": pair_group,
                    "audio_asset_id_sha256": audio_source["asset_id_sha256"],
                    "video_asset_id_sha256": video_source["asset_id_sha256"],
                    "audio_feature_record": audio_source["feature_record"],
                    "video_feature_record": video_source["feature_record"],
                    "target": target,
                }
            )
    return output, leftovers
