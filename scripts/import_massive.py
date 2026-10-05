from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any

from tiny_omni_decision.massive import build_massive_splits, partition_from_raw_line

ARCHIVE_SHA256 = "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8"
ARCHIVE_MEMBER = "1.0/data/en-US.jsonl"
SOURCE_REVISION = "f966f21846043aabef9b0f974fa7970027f43738"
PARTITION_RE = re.compile(rb'"partition"\s*:\s*"(train|dev|test)"')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import only en-US MASSIVE train/dev rows; never parse reserved test labels."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/teacher-quality/massive/amazon-massive-dataset-1.0.tar.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/teacher-quality-next/massive"),
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    actual_archive_hash = sha256_file(args.archive)
    if actual_archive_hash != ARCHIVE_SHA256:
        raise SystemExit(f"MASSIVE archive SHA256 mismatch: {actual_archive_hash}")

    rows: list[dict[str, Any]] = []
    with tarfile.open(args.archive, "r:gz") as archive:
        member = archive.extractfile(ARCHIVE_MEMBER)
        if member is None:
            raise SystemExit(f"archive member not found: {ARCHIVE_MEMBER}")
        for raw in member:
            # Inspect only the partition marker before deciding whether to parse a record.
            partition = partition_from_raw_line(raw)
            if partition == "test":
                continue
            row = json.loads(raw)
            if row.get("partition") != partition:
                raise SystemExit("partition marker changed between prefilter and row parsing")
            rows.append(row)

    train, validation, accounting = build_massive_splits(
        rows, revision=SOURCE_REVISION, seed=args.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.jsonl"
    validation_path = args.output / "validation.jsonl"
    train_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in train), encoding="utf-8"
    )
    validation_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in validation), encoding="utf-8"
    )
    metadata = {
        "dataset": "alexa/massive",
        "revision": SOURCE_REVISION,
        "archive_sha256": actual_archive_hash,
        "archive_member": ARCHIVE_MEMBER,
        "locale": "en-US",
        "seed": args.seed,
        "test_partition_parsed": False,
        "normalization": (
            "Unicode NFKC, casefold, collapse whitespace; normalized utterances stay grouped."
        ),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "accounting": accounting,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
