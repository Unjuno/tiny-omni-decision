"""Screen CATER video IDs against already-used Clevr-4 image indices."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

CATER_ROW = re.compile(r"^(CATER_new_(\d+)\.avi)\s+(\d+)$")
CLEVR_ASSET = re.compile(r"^CLEVR_new_(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cater_list(path: Path, *, num_classes: int = 64) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = CATER_ROW.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"{path}:{line_number}: malformed CATER row")
        video_id, label = int(match.group(2)), int(match.group(3))
        if video_id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate CATER video id {video_id}")
        if not 0 <= label < num_classes:
            raise ValueError(f"{path}:{line_number}: label outside [0, {num_classes})")
        seen_ids.add(video_id)
        rows.append({"video_id": video_id, "label": label})
    return rows


def read_clevr4_indices(path: Path) -> set[int]:
    indices: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        match = CLEVR_ASSET.fullmatch(str(row.get("asset_id", "")))
        if match is None:
            raise ValueError(f"{path}:{line_number}: invalid Clevr-4 asset_id")
        index = int(match.group(1))
        if index in indices:
            raise ValueError(f"{path}:{line_number}: duplicate Clevr-4 asset index {index}")
        indices.add(index)
    return indices


def audit_split(rows: list[dict[str, int]], used_indices: set[int]) -> dict[str, Any]:
    collisions = [row for row in rows if row["video_id"] in used_indices]
    kept = [row for row in rows if row["video_id"] not in used_indices]
    counts = Counter(row["label"] for row in kept)
    return {
        "rows": len(rows),
        "index_collisions": len(collisions),
        "kept_after_conservative_exclusion": len(kept),
        "excluded_video_ids": [row["video_id"] for row in collisions],
        "kept_per_class": {str(label): counts[label] for label in range(64)},
        "minimum_kept_per_class": min(counts.values(), default=0),
        "missing_classes": [label for label in range(64) if not counts[label]],
    }


def build_report(
    *, train_list: Path, selector_list: Path, clevr4_manifests: list[Path]
) -> dict[str, Any]:
    used_indices: set[int] = set()
    manifest_receipts = []
    for manifest in clevr4_manifests:
        indices = read_clevr4_indices(manifest)
        used_indices.update(indices)
        manifest_receipts.append(
            {
                "path": str(manifest),
                "sha256": sha256_file(manifest),
                "rows": len(indices),
            }
        )
    return {
        "schema_version": 1,
        "purpose": "conservative numeric asset-index collision screen; not content deduplication",
        "clevr4_used_numeric_indices": len(used_indices),
        "clevr4_manifests": manifest_receipts,
        "cater_train_subset_t": {
            "path": str(train_list),
            "sha256": sha256_file(train_list),
            **audit_split(read_cater_list(train_list), used_indices),
        },
        "cater_selector_subset_v": {
            "path": str(selector_list),
            "sha256": sha256_file(selector_list),
            **audit_split(read_cater_list(selector_list), used_indices),
        },
        "limitations": [
            "numeric suffix collisions are conservatively excluded, not asserted duplicate media",
            "different IDs with identical rendered content are not detected",
            "no video or sealed-audit data is read",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--selector-list", type=Path, required=True)
    parser.add_argument("--clevr4-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        train_list=args.train_list,
        selector_list=args.selector_list,
        clevr4_manifests=args.clevr4_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
