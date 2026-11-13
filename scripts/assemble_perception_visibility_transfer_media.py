"""Assemble a verified hard-linked media view across fixed transfer attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tiny_omni_decision.perception_visibility_transfer import merge_media_receipts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def assemble(
    cohort_path: Path,
    reuse_root: Path,
    reuse_receipt_path: Path,
    new_root: Path,
    new_receipt_path: Path,
) -> dict[str, Any]:
    cohort = read_json(cohort_path)
    prior_receipt = read_json(reuse_receipt_path)
    new_receipt = read_json(new_receipt_path)
    if prior_receipt.get("status") != "PASS_TRAIN_MEDIA_ACQUISITION":
        raise ValueError("prior acquisition receipt is not complete")
    if new_receipt.get("status") != "PASS_TRAIN_MEDIA_ACQUISITION":
        raise ValueError("new acquisition receipt is not complete")
    video_ids = {row["video_id"] for row in cohort["examples"]}
    rows = merge_media_receipts(
        video_ids, prior_receipt.get("files", []), new_receipt.get("files", [])
    )
    for row in rows:
        video_id = str(row["video_id"])
        suffix = Path(str(row["member"])).suffix
        new_path = new_root / "train" / f"{video_id}{suffix}"
        source_root = new_root if new_path.is_file() else reuse_root
        source = source_root / "train" / f"{video_id}{suffix}"
        destination = new_root / "train" / f"{video_id}{suffix}"
        if not source.is_file():
            raise FileNotFoundError(source)
        if sha256_file(source) != row["sha256"]:
            raise ValueError(f"media hash differs from acquisition receipt: {source}")
        if source != destination:
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite media: {destination}")
            os.link(source, destination)
        row["output_path"] = str(destination)
        if destination.stat().st_size != int(row["expanded_bytes"]):
            raise ValueError(f"media size differs from acquisition receipt: {destination}")
    output = new_root / "acquisition_manifest.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite media index: {output}")
    merged_receipt = {
        "status": "PASS_MERGED_TRAIN_MEDIA_INDEX",
        "source_split": "official train only",
        "cohort_sha256": sha256_file(cohort_path),
        "prior_acquisition_sha256": sha256_file(reuse_receipt_path),
        "new_acquisition_sha256": sha256_file(new_receipt_path),
        "files": rows,
        "unique_video_count": len(video_ids),
        "file_count": len(rows),
        "test_split_opened": False,
    }
    output.write_text(json.dumps(merged_receipt, indent=2) + "\n", encoding="utf-8")
    return {
        "status": merged_receipt["status"],
        "unique_video_count": len(video_ids),
        "file_count": len(rows),
        "media_index_sha256": sha256_file(output),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, required=True)
    parser.add_argument("--reuse-receipt", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--new-receipt", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            assemble(
                args.cohort.resolve(strict=True),
                args.reuse_root.resolve(strict=True),
                args.reuse_receipt.resolve(strict=True),
                args.new_root.resolve(strict=True),
                args.new_receipt.resolve(strict=True),
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
