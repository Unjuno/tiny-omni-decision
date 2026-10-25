"""Filter pilot tasks to records with matching current Flickr page declarations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def filter_to_page_matches(
    manifest: dict[str, Any],
    receipts: list[dict[str, Any]],
    *,
    expected_split: str,
    receipt_sha256: str,
) -> dict[str, Any]:
    if expected_split not in {"train", "validation"}:
        raise ValueError("expected_split must be train or validation")
    if manifest.get("split") != expected_split:
        raise ValueError("task manifest has the wrong split")
    by_id: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        image_id = receipt.get("ImageID")
        if not isinstance(image_id, str) or not image_id or image_id in by_id:
            raise ValueError("receipts must contain unique nonempty ImageID values")
        if receipt.get("Subset") != expected_split:
            raise ValueError("receipt has the wrong split")
        by_id[image_id] = receipt

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected_count = 0
    for example in manifest.get("examples", []):
        image_id = example.get("image_id")
        if not isinstance(image_id, str) or image_id in seen:
            raise ValueError("manifest examples must have unique image IDs")
        seen.add(image_id)
        if image_id not in by_id:
            raise ValueError("task example is missing its private attribution receipt")
        if by_id[image_id].get("page_license_and_attribution_match") is True:
            kept.append(example)
        else:
            rejected_count += 1
    if not kept:
        raise ValueError("no page-matched examples remain")

    counts = Counter(example["target_label"] for example in kept)
    target_labels = tuple(manifest.get("target_labels", ()))
    return manifest | {
        "examples": kept,
        "example_count": len(kept),
        "unique_image_count": len({row["image_id"] for row in kept}),
        "counts_by_target": {label: counts.get(label, 0) for label in sorted(target_labels)},
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(row["sample_id"] for row in kept).encode("utf-8")
        ).hexdigest(),
        "page_check_receipt_sha256": receipt_sha256,
        "page_check_gate": "flickr CC BY 2.0 declaration and attribution profile match",
        "excluded_nonmatching_or_unavailable_count": rejected_count,
        "individual_rights_verified": False,
        "independent_ownership_verification_performed": False,
        "media_downloaded": False,
        "model_loaded": False,
    }


def build_page_matched_pair(
    train_manifest: dict[str, Any],
    validation_manifest: dict[str, Any],
    train_receipts: list[dict[str, Any]],
    validation_receipts: list[dict[str, Any]],
    *,
    train_receipt_sha256: str,
    validation_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    train = filter_to_page_matches(
        train_manifest,
        train_receipts,
        expected_split="train",
        receipt_sha256=train_receipt_sha256,
    )
    validation = filter_to_page_matches(
        validation_manifest,
        validation_receipts,
        expected_split="validation",
        receipt_sha256=validation_receipt_sha256,
    )
    train_rows = {
        row["ImageID"]: row
        for row in train_receipts
        if row.get("page_license_and_attribution_match") is True
    }
    validation_rows = {
        row["ImageID"]: row
        for row in validation_receipts
        if row.get("page_license_and_attribution_match") is True
    }
    train_ids = {example["image_id"] for example in train["examples"]}
    validation_ids = {example["image_id"] for example in validation["examples"]}
    if train_ids & validation_ids:
        raise ValueError("page-matched ImageID overlap across splits")
    train_hashes = {train_rows[image_id]["OriginalMD5"] for image_id in train_ids}
    validation_hashes = {validation_rows[image_id]["OriginalMD5"] for image_id in validation_ids}
    if train_hashes & validation_hashes:
        raise ValueError("page-matched OriginalMD5 overlap across splits")
    summary = {
        "train_examples": len(train_ids),
        "validation_examples": len(validation_ids),
        "train_validation_image_id_overlap": 0,
        "train_validation_original_md5_overlap": 0,
        "individual_rights_verified": False,
        "media_downloaded": False,
        "model_loaded": False,
    }
    return train, validation, summary


def _write_new(path: Path, data: dict[str, Any]) -> None:
    resolved = path.resolve()
    if resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("source-identified task manifests must be stored outside the repository")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--train-receipts", type=Path, required=True)
    parser.add_argument("--validation-receipts", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    train, validation, summary = build_page_matched_pair(
        json.loads(args.train_manifest.read_text(encoding="utf-8")),
        json.loads(args.validation_manifest.read_text(encoding="utf-8")),
        read_jsonl(args.train_receipts),
        read_jsonl(args.validation_receipts),
        train_receipt_sha256=sha256_file(args.train_receipts),
        validation_receipt_sha256=sha256_file(args.validation_receipts),
    )
    destinations = tuple(
        path.resolve()
        for path in (args.train_output, args.validation_output, args.summary_output)
    )
    if len(set(destinations)) != len(destinations):
        raise ValueError("output paths must be distinct")
    if any(path.is_relative_to(REPOSITORY_ROOT) for path in destinations):
        raise ValueError("source-identified task manifests must be stored outside the repository")
    if any(path.exists() for path in destinations):
        raise FileExistsError("refusing to overwrite an existing output")
    _write_new(args.train_output, train)
    _write_new(args.validation_output, validation)
    _write_new(args.summary_output, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
