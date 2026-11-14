"""Fail-closed verification for the frozen FSD50K annotation-decision media."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.build_fsd50k_annotation_decision_manifest import (
    ALLOWED_LICENSES,
    DATASET_REVISION,
    canonical_sha256,
    sha256_file,
    summarize_split,
)

SPLITS = ("train", "calibration", "validation")


def load_frozen_manifest(
    preflight_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    report = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "PREFLIGHT_PASS"
        or report.get("dataset_revision") != DATASET_REVISION
    ):
        raise ValueError("FSD50K preflight is not a passing pinned manifest")
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        rows = [
            json.loads(line)
            for line in (preflight_dir / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if summarize_split(rows) != report["splits"][split]:
            raise ValueError(f"frozen FSD50K {split} summary mismatch")
        splits[split] = rows
    expected_manifest_hash = canonical_sha256(
        {"splits": report["splits"], "source_sha256": report["source_sha256"]}
    )
    if report.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("frozen FSD50K manifest hash mismatch")
    return report, splits


def sha256_and_crc32(path: Path) -> tuple[int, str, int]:
    digest = hashlib.sha256()
    crc = 0
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
            crc = zlib.crc32(block, crc)
    return size, digest.hexdigest(), crc


def validate_acquisition(preflight_dir: Path, media_dir: Path) -> dict[str, Any]:
    preflight, splits = load_frozen_manifest(preflight_dir)
    result_path = media_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PASS"
        or result.get("dataset_revision") != DATASET_REVISION
        or result.get("preflight_manifest_sha256") != preflight["manifest_sha256"]
    ):
        raise ValueError("complete acquisition for the exact frozen manifest is required")
    if result.get("official_eval_labels_read") is not False or result.get(
        "official_eval_media_acquired"
    ) is not False:
        raise ValueError("official FSD50K eval must remain unopened")

    expected: dict[str, tuple[str, dict[str, Any]]] = {}
    source_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for split, rows in splits.items():
        for row in rows:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.isdecimal():
                raise ValueError("invalid FSD50K sample ID")
            if sample_id in expected:
                raise ValueError("duplicate sample ID across frozen splits")
            expected[sample_id] = (split, row)
            source_splits[row["source_id"]].add(split)
            group_splits[row["group_id"]].add(split)
    if any(len(owners) > 1 for owners in source_splits.values()):
        raise ValueError("source ID overlap across FSD50K splits")
    if any(len(owners) > 1 for owners in group_splits.values()):
        raise ValueError("uploader overlap across FSD50K splits")

    receipt_path = media_dir / "files.jsonl"
    if result.get("files_jsonl_sha256") != sha256_file(receipt_path):
        raise ValueError("acquisition receipt hash mismatch")
    receipts = [
        json.loads(line)
        for line in receipt_path.read_text(encoding="utf-8").splitlines()
    ]
    if (
        len(receipts) != len(expected)
        or result.get("expected_files") != len(expected)
        or result.get("downloaded_files") != len(expected)
    ):
        raise ValueError("acquisition receipt/file count mismatch")

    seen_ids: set[str] = set()
    content_splits: dict[str, set[str]] = defaultdict(set)
    verified_by_split = {split: 0 for split in SPLITS}
    for receipt in receipts:
        sample_id = receipt.get("sample_id")
        if sample_id in seen_ids or sample_id not in expected:
            raise ValueError("duplicate or unexpected sample in acquisition receipt")
        seen_ids.add(sample_id)
        split, manifest_row = expected[sample_id]
        for field in ("source_id", "group_id", "license", "split"):
            if receipt.get(field) != manifest_row[field]:
                raise ValueError(f"acquisition receipt {field} mismatch: {sample_id}")
        if manifest_row["license"] not in ALLOWED_LICENSES:
            raise ValueError(f"unapproved FSD50K clip license: {sample_id}")

        path = media_dir / split / f"{sample_id}.wav"
        if not path.is_file():
            raise ValueError(f"acquired FSD50K media missing: {sample_id}")
        size, digest, crc = sha256_and_crc32(path)
        if (
            size != receipt.get("bytes")
            or digest != receipt.get("sha256")
            or crc != receipt.get("zip_crc32")
        ):
            raise ValueError(f"acquired FSD50K media integrity mismatch: {sample_id}")
        content_splits[digest].add(split)
        verified_by_split[split] += 1

    if len(seen_ids) != len(expected):
        raise ValueError("acquisition is missing frozen FSD50K IDs")
    if any(len(owners) > 1 for owners in content_splits.values()):
        raise ValueError("exact media bytes overlap across FSD50K splits")

    expected_paths = {
        media_dir / split / f"{sample_id}.wav" for sample_id, (split, _) in expected.items()
    }
    actual_paths = set(media_dir.rglob("*.wav"))
    if actual_paths != expected_paths:
        raise ValueError("unexpected or missing WAV files in acquisition directory")

    return {
        "status": "PASS",
        "dataset_revision": DATASET_REVISION,
        "preflight_manifest_sha256": preflight["manifest_sha256"],
        "acquisition_result_sha256": sha256_file(result_path),
        "receipt_sha256": sha256_file(receipt_path),
        "verified_files": len(seen_ids),
        "verified_files_by_split": verified_by_split,
        "unique_media_sha256": len(content_splits),
        "cross_split_exact_media_overlap": 0,
        "official_eval_labels_read": False,
        "official_eval_media_acquired": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("C:/CodexArtifacts/pretrained-reuse-r2")
    parser.add_argument(
        "--preflight",
        type=Path,
        default=root / "fsd50k-annotation-presence-v1-preflight-20261115-attempt-01",
    )
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_acquisition(args.preflight, args.media)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite verification report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
