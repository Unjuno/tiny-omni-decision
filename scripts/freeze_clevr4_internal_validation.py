"""Freeze an exposure-screened Clevr-4 internal validation pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.audit_clevr4_content_overlap import _load_jsonl, sha256_file


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _order_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(("\n".join(row["id"] for row in rows) + "\n").encode()).hexdigest()


def freeze_pool(
    *,
    labels_path: Path,
    media_manifest_path: Path,
    prior_image_dirs: list[Path],
    near_pair_result_path: Path,
    output: Path,
    selector_count: int = 1000,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    labels = _load_jsonl(labels_path)
    media = _load_jsonl(media_manifest_path)
    if len(labels) != len(media) or not labels:
        raise ValueError("labels/media count mismatch or empty candidate pool")
    if [row["id"] for row in labels] != [row["id"] for row in media]:
        raise ValueError("labels and media manifest ID order differ")
    if any(row.get("source_split") != "train" for row in labels):
        raise ValueError("candidate pool includes non-training source records")
    near_result = json.loads(near_pair_result_path.read_text(encoding="utf-8"))
    if near_result["candidate_manifest_sha256"] != sha256_file(media_manifest_path):
        raise ValueError("near-pair audit was run against a different media manifest")
    near_ids = {row["candidate_id"] for row in near_result["near_pairs"]}
    prior_ids = {path.stem for directory in prior_image_dirs for path in directory.glob("*.png")}
    ids = {row["id"] for row in labels}
    if len(ids) != len(labels):
        raise ValueError("duplicate candidate IDs")
    quarantined = ids & (near_ids | prior_ids)
    eligible_labels = [row for row in labels if row["id"] not in quarantined]
    eligible_media = [row for row in media if row["id"] not in quarantined]
    if selector_count <= 0 or selector_count >= len(eligible_labels):
        raise ValueError("selector_count must leave nonempty selector and report pools")
    ranked = sorted(
        eligible_labels,
        key=lambda row: hashlib.sha256(b"clevr4-selector-v1:" + row["id"].encode()).digest(),
    )
    selector_ids = {row["id"] for row in ranked[:selector_count]}
    selector_labels = [row for row in eligible_labels if row["id"] in selector_ids]
    report_labels = [row for row in eligible_labels if row["id"] not in selector_ids]
    selector_media = [row for row in eligible_media if row["id"] in selector_ids]
    report_media = [row for row in eligible_media if row["id"] not in selector_ids]

    output.mkdir(parents=True, exist_ok=False)
    files = {
        "selector-labels.jsonl": selector_labels,
        "selector-media-manifest.jsonl": selector_media,
        "report-labels.jsonl": report_labels,
        "report-media-manifest.jsonl": report_media,
        "quarantined-labels.jsonl": [row for row in labels if row["id"] in quarantined],
        "quarantined-media-manifest.jsonl": [row for row in media if row["id"] in quarantined],
    }
    for name, rows in files.items():
        _write_jsonl(output / name, rows)
    receipt = {
        "status": "FROZEN_EXPOSURE_SCREENED_INTERNAL_VALIDATION_POOL",
        "rule": (
            "quarantine prior 10k ID collisions or nearest pHash <= 4; "
            "this conservative rule is not proof of identity"
        ),
        "selector_rule": (
            "rank by SHA256('clevr4-selector-v1:' + ID); take the first "
            "selector_count IDs; preserve source order in files"
        ),
        "selector_count": len(selector_labels),
        "report_count": len(report_labels),
        "quarantined_count": len(quarantined),
        "quarantined_by_near_phash_only": len((near_ids - prior_ids) & ids),
        "quarantined_by_prior_id_only": len((prior_ids - near_ids) & ids),
        "quarantined_by_both": len(prior_ids & near_ids & ids),
        "candidate_count": len(labels),
        "candidate_labels_sha256": sha256_file(labels_path),
        "candidate_media_manifest_sha256": sha256_file(media_manifest_path),
        "near_pair_audit_sha256": sha256_file(near_pair_result_path),
        "selector_labels_sha256": sha256_file(output / "selector-labels.jsonl"),
        "selector_media_manifest_sha256": sha256_file(output / "selector-media-manifest.jsonl"),
        "report_labels_sha256": sha256_file(output / "report-labels.jsonl"),
        "report_media_manifest_sha256": sha256_file(output / "report-media-manifest.jsonl"),
        "quarantined_labels_sha256": sha256_file(output / "quarantined-labels.jsonl"),
        "quarantined_media_manifest_sha256": sha256_file(
            output / "quarantined-media-manifest.jsonl"
        ),
        "selector_id_order_sha256": _order_hash(selector_labels),
        "report_id_order_sha256": _order_hash(report_labels),
        "source_files_modified": False,
        "media_files_written": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--prior-image-dir", type=Path, action="append", required=True)
    parser.add_argument("--near-pair-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selector-count", type=int, default=1000)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_pool(
                labels_path=args.labels,
                media_manifest_path=args.media_manifest,
                prior_image_dirs=args.prior_image_dir,
                near_pair_result_path=args.near_pair_result,
                output=args.output,
                selector_count=args.selector_count,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
