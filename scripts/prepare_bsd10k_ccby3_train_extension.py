"""Prepare a CC-BY-3.0-only training addition without moving held-out groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
try:
    from scripts.audit_bsd10k_metadata import sha256_file
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from audit_bsd10k_metadata import sha256_file

EXPECTED_RECORD_ID = "19868804"
EXPECTED_METADATA_SHA256 = "c090b97e5cdac221f867d5a970b0a8326833e57cfe586c5886494170b25413f1"
EXPECTED_BASE_SPLIT_SHA256 = "e1ae3d4088b792619a08a87023e4aeb2b1b5215091ba9772c0dc47fe88842c33"
EXPECTED_CC_BY3_COUNT = 2156
EXPECTED_HELDOUT_GROUP_EXCLUSIONS = 8
EXPECTED_EXTENSION_COUNT = 2148
EXPECTED_EXTENSION_GROUPS = 431
EXPECTED_NEW_GROUPS = 405
CC_BY3_LICENSE = "CC-BY-3.0"
CC_BY3_LICENSE_URL = "https://creativecommons.org/licenses/by/3.0/"
BSD10K_CLASSES = {
    "fx-a",
    "fx-el",
    "fx-ex",
    "fx-h",
    "fx-m",
    "fx-n",
    "fx-o",
    "fx-v",
    "is-e",
    "is-k",
    "is-p",
    "is-s",
    "is-w",
    "m-m",
    "m-si",
    "m-sp",
    "sp-c",
    "sp-p",
    "sp-s",
    "ss-i",
    "ss-n",
    "ss-s",
    "ss-u",
}


def _canonical_path(value: str) -> str:
    return "/".join(part for part in urlparse(value.strip()).path.casefold().split("/") if part)


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing extension artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def prepare_extension(
    metadata_zip: Path,
    base_split_manifest: Path,
    output_manifest: Path,
    additions_manifest: Path,
    group_roles_json: Path,
    attribution_csv: Path,
    report_json: Path,
) -> dict:
    metadata_sha = sha256_file(metadata_zip)
    base_sha = sha256_file(base_split_manifest)
    if metadata_sha != EXPECTED_METADATA_SHA256:
        raise ValueError("BSD10k metadata archive differs from the pinned v1.2 source")
    if base_sha != EXPECTED_BASE_SPLIT_SHA256:
        raise ValueError("base split differs from the frozen seed-17 manifest")
    outputs = {output_manifest, additions_manifest, group_roles_json, attribution_csv, report_json}
    if len(outputs) != 5:
        raise ValueError("all extension outputs must be distinct")

    base = json.loads(base_split_manifest.read_text(encoding="utf-8"))
    if (
        base.get("schema") != "tiny-omni-decision/bsd10k-group-split-v1"
        or base.get("source_record_id") != EXPECTED_RECORD_ID
        or base.get("metadata_zip_sha256") != metadata_sha
    ):
        raise ValueError("base split manifest does not match the pinned BSD10k source")
    base_rows = base.get("examples")
    if not isinstance(base_rows, list) or not base_rows:
        raise ValueError("base split manifest has no examples")

    group_splits: dict[str, str] = {}
    existing_ids: set[str] = set()
    for row in base_rows:
        sound_id = str(row.get("sound_id", ""))
        group = str(row.get("uploader_group_sha256", ""))
        split = str(row.get("split", ""))
        if not sound_id.isdecimal() or sound_id in existing_ids:
            raise ValueError("base split has malformed or duplicate sound IDs")
        if len(group) != 64 or split not in {"train", "validation", "test"}:
            raise ValueError("base split has malformed group or split data")
        existing_ids.add(sound_id)
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError("base split leaks an uploader group across partitions")
    heldout_groups = {group for group, split in group_splits.items() if split != "train"}
    base_train_groups = {group for group, split in group_splits.items() if split == "train"}
    from tiny_omni_decision.textfree_audio import speaker_group_split

    base_train_group_order = [
        row["uploader_group_sha256"] for row in base_rows if row["split"] == "train"
    ]
    _, base_calibration_indices = speaker_group_split(
        base_train_group_order, seed=17, calibration_fraction=0.2
    )
    frozen_calibration_groups = sorted(
        {base_train_group_order[index] for index in base_calibration_indices}
    )

    additions: list[dict] = []
    attributions: list[dict[str, str]] = []
    excluded_groups: set[str] = set()
    heldout_exclusions = 0
    seen_ids: set[str] = set()
    class_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    cc_by3_count = 0
    with zipfile.ZipFile(metadata_zip) as archive:
        csv_names = [
            name for name in archive.namelist() if name.casefold().endswith("bsd10k_metadata.csv")
        ]
        if len(csv_names) != 1:
            raise ValueError("metadata archive must contain exactly one BSD10k metadata CSV")
        with archive.open(csv_names[0]) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            required = {"sound_id", "class", "confidence", "uploader", "license", "title"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("metadata is missing required attribution or label fields")
            for row in reader:
                if _canonical_path(row["license"]) != "licenses/by/3.0":
                    continue
                cc_by3_count += 1
                sound_id = row["sound_id"].strip()
                uploader = row["uploader"].strip()
                normalized_uploader = uploader.casefold()
                group_hash = hashlib.sha256(normalized_uploader.encode("utf-8")).hexdigest()
                if group_hash in heldout_groups:
                    heldout_exclusions += 1
                    excluded_groups.add(group_hash)
                    continue
                label = row["class"].strip()
                confidence = row["confidence"].strip()
                if (
                    not sound_id.isdecimal()
                    or sound_id in existing_ids
                    or sound_id in seen_ids
                    or not uploader
                    or label not in BSD10K_CLASSES
                    or confidence not in {"1", "2", "3", "4", "5"}
                ):
                    raise ValueError("eligible CC-BY-3.0 training metadata is malformed")
                seen_ids.add(sound_id)
                additions.append(
                    {
                        "sound_id": sound_id,
                        "class": label,
                        "confidence": int(confidence),
                        "license": CC_BY3_LICENSE,
                        "uploader_group_sha256": group_hash,
                        "split": "train",
                    }
                )
                class_counts[label] += 1
                confidence_counts[confidence] += 1
                attributions.append(
                    {
                        "sound_id": sound_id,
                        "title": row["title"].strip(),
                        "creator_uploader": uploader,
                        "source_url": f"https://freesound.org/s/{sound_id}/",
                        "license_url": CC_BY3_LICENSE_URL,
                        "changes": (
                            "audio retained as supplied; fixed-rate resampling is applied only "
                            "during feature extraction"
                        ),
                    }
                )

    if cc_by3_count != EXPECTED_CC_BY3_COUNT:
        raise ValueError("CC-BY-3.0 source record count changed; re-audit before proceeding")
    if heldout_exclusions != EXPECTED_HELDOUT_GROUP_EXCLUSIONS:
        raise ValueError("held-out-group exclusion count changed; re-audit before proceeding")
    if len(additions) != EXPECTED_EXTENSION_COUNT:
        raise ValueError("eligible training addition count changed; re-audit before proceeding")
    all_added_groups = {row["uploader_group_sha256"] for row in additions}
    new_groups = all_added_groups - base_train_groups
    if len(all_added_groups) != EXPECTED_EXTENSION_GROUPS or len(new_groups) != EXPECTED_NEW_GROUPS:
        raise ValueError("eligible uploader-group counts changed; re-audit before proceeding")
    if all_added_groups & heldout_groups:
        raise ValueError("training extension overlaps a frozen validation/test uploader")

    combined = dict(base)
    combined["examples"] = sorted(base_rows + additions, key=lambda row: int(row["sound_id"]))
    combined["training_extension"] = {
        "license": CC_BY3_LICENSE,
        "metadata_sha256": metadata_sha,
        "base_split_manifest_sha256": base_sha,
        "added_example_count": len(additions),
        "added_uploader_group_count": len(all_added_groups),
        "new_uploader_group_count": len(new_groups),
        "validation_and_test_rows_preserved_byte_for_byte": True,
    }
    report = {
        "schema": "tiny-omni-decision/bsd10k-ccby3-train-extension-preflight-v1",
        "status": "PASS_METADATA_ONLY_PREFLIGHT",
        "source_record_id": EXPECTED_RECORD_ID,
        "metadata_zip_sha256": metadata_sha,
        "base_split_manifest_sha256": base_sha,
        "cc_by3_total_records": cc_by3_count,
        "excluded_records_in_frozen_validation_or_test_uploader_groups": heldout_exclusions,
        "excluded_heldout_group_count": len(excluded_groups),
        "added_training_records": len(additions),
        "added_uploader_groups": len(all_added_groups),
        "new_uploader_groups": len(new_groups),
        "records_in_existing_train_uploader_groups": sum(
            group_hash in base_train_groups
            for group_hash in (row["uploader_group_sha256"] for row in additions)
        ),
        "train_examples_by_class": dict(sorted(class_counts.items())),
        "train_examples_by_annotation_confidence": dict(sorted(confidence_counts.items())),
        "validation_and_test_assignments_changed": False,
        "custom_test_or_sealed_audit_loaded": False,
        "media_downloaded_or_hashed": False,
    }
    manifest_text = json.dumps(combined, indent=2, sort_keys=True) + "\n"
    additions_document = {
        **{key: value for key, value in base.items() if key != "examples"},
        "examples": additions,
        "training_extension": combined["training_extension"],
    }
    additions_text = json.dumps(additions_document, indent=2, sort_keys=True) + "\n"
    group_roles = {
        "schema": "tiny-omni-decision/bsd10k-frozen-calibration-groups-v1",
        "seed": 17,
        "calibration_fraction": 0.2,
        "source_base_split_manifest_sha256": base_sha,
        "calibration_uploader_group_sha256": frozen_calibration_groups,
    }
    group_roles_text = json.dumps(group_roles, indent=2, sort_keys=True) + "\n"
    attribution_stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        attribution_stream,
        fieldnames=[
            "sound_id",
            "title",
            "creator_uploader",
            "source_url",
            "license_url",
            "changes",
        ],
    )
    writer.writeheader()
    writer.writerows(sorted(attributions, key=lambda item: int(item["sound_id"])))
    report["combined_split_manifest_sha256"] = hashlib.sha256(
        manifest_text.encode("utf-8")
    ).hexdigest()
    report["additions_manifest_sha256"] = hashlib.sha256(additions_text.encode("utf-8")).hexdigest()
    report["frozen_group_roles_sha256"] = hashlib.sha256(
        group_roles_text.encode("utf-8")
    ).hexdigest()
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _write_new(output_manifest, manifest_text)
    _write_new(additions_manifest, additions_text)
    _write_new(group_roles_json, group_roles_text)
    _write_new(attribution_csv, attribution_stream.getvalue())
    _write_new(report_json, report_text)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--base-split-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--additions-manifest", type=Path, required=True)
    parser.add_argument("--group-roles-json", type=Path, required=True)
    parser.add_argument("--attribution-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_extension(
        args.metadata_zip.resolve(strict=True),
        args.base_split_manifest.resolve(strict=True),
        args.output_manifest.resolve(),
        args.additions_manifest.resolve(),
        args.group_roles_json.resolve(),
        args.attribution_csv.resolve(),
        args.report_json.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
