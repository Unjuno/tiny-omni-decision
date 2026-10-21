"""Create local-path-only extractor manifests without changing canonical v2 samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prepare_feature_input(
    *,
    sample_dir: Path,
    old_sample_dir: Path,
    expanded_media_root: Path,
    output_dir: Path,
    include_old_v1: bool = True,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature input directory: {output_dir}")
    canonical = {
        split: _read_rows(sample_dir / f"{split}.jsonl") for split in ("train", "validation")
    }
    old_hashes = {
        str(row["media_sha256"])
        for split in ("train", "validation")
        for row in _read_rows(old_sample_dir / f"{split}.jsonl")
    }
    rewritten: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    seen_hashes: set[str] = set()
    for split, rows in canonical.items():
        for original in rows:
            row = dict(original)
            media_hash = str(row["media_sha256"])
            if not include_old_v1 and media_hash in old_hashes:
                continue
            media_root = old_sample_dir if media_hash in old_hashes else expanded_media_root
            media_path = (media_root / str(row["media_path"])).resolve(strict=True)
            if media_path.stat().st_size <= 0 or sha256_file(media_path) != media_hash:
                raise ValueError(f"media identity mismatch: {media_path}")
            row["media_path"] = str(media_path)
            rewritten[split].append(row)
            seen_hashes.add(media_hash)
    if not rewritten["train"] or not rewritten["validation"]:
        raise ValueError("both v2 sample splits are required for feature extraction")
    output_dir.mkdir(parents=True)
    for split, rows in rewritten.items():
        with (output_dir / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    sample_hashes = {
        name: sha256_file(sample_dir / name)
        for name in ("train.jsonl", "validation.jsonl", "fetch-report.json")
    }
    report = {
        "status": "complete_clevrer_synthetic_av_v2_local_feature_input",
        "canonical_sample_manifest_sha256": sample_hashes,
        "feature_input_manifest_sha256": {
            split: sha256_file(output_dir / f"{split}.jsonl")
            for split in ("train", "validation")
        },
        "unique_media_count": len(seen_hashes),
        "train_unique_media_count": len(
            {str(row["media_sha256"]) for row in rewritten["train"]}
        ),
        "validation_unique_media_count": len(
            {str(row["media_sha256"]) for row in rewritten["validation"]}
        ),
        "all_media_hashes_reverified": True,
        "includes_old_v1_media": include_old_v1,
        "validation_test_or_sealed_data_opened": False,
    }
    (output_dir / "fetch-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--old-sample-dir", type=Path, required=True)
    parser.add_argument("--expanded-media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--new-scenes-only",
        action="store_true",
        help="omit old v1 media; useful when extracting only the new v2 scene groups",
    )
    args = parser.parse_args()
    report = prepare_feature_input(
        sample_dir=args.sample_dir.resolve(strict=True),
        old_sample_dir=args.old_sample_dir.resolve(strict=True),
        expanded_media_root=args.expanded_media_root.resolve(strict=True),
        output_dir=args.output_dir.resolve(),
        include_old_v1=not args.new_scenes_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
