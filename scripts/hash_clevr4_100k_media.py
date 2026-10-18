"""Hash only indexed official-train Clevr-4 images without extracting them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def hash_indexed_media(
    archive: Path,
    index: Path,
    previous_manifest: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    train = [
        json.loads(line) for line in (index / "train-media-manifest.jsonl").read_text().splitlines()
    ]
    validation = [
        json.loads(line)
        for line in (index / "validation-media-manifest.jsonl").read_text().splitlines()
    ]
    previous = [json.loads(line) for line in previous_manifest.read_text().splitlines()]
    previous_hashes = {row["sha256"].lower() for row in previous}
    if len(previous_hashes) != len(previous):
        raise ValueError("previous manifest contains duplicate media hashes")

    with ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        for rows in (train, validation):
            for row in rows:
                member = row["archive_member"]
                if member not in names:
                    raise ValueError(f"indexed media is absent from archive: {member}")
                with zipped.open(member) as stream:
                    row["media_sha256"] = sha256_stream(stream)

    train_hashes = {row["media_sha256"] for row in train}
    validation_hashes = {row["media_sha256"] for row in validation}
    overlap = train_hashes & validation_hashes
    if overlap:
        raise ValueError(f"train/validation media overlap: {len(overlap)}")
    previous_overlap = (train_hashes | validation_hashes) & previous_hashes
    output.mkdir(parents=True)
    manifest_hashes = {}
    for name, rows in (("train", train), ("validation", validation)):
        manifest_path = output / f"{name}-media-manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest_hashes[manifest_path.name] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result = {
        "status": "MEDIA_HASHED",
        "archive": str(archive),
        "indexed_train_examples": len(train),
        "indexed_validation_examples": len(validation),
        "unique_train_media_hashes": len(train_hashes),
        "unique_validation_media_hashes": len(validation_hashes),
        "train_validation_media_overlap": len(overlap),
        "previous_train_manifest_examples": len(previous),
        "overlap_with_previous_train_media": len(previous_overlap),
        "overlap_hashes": sorted(previous_overlap),
        "media_manifest_sha256": manifest_hashes,
        "official_val_media_opened": 0,
        "files_extracted": 0,
    }
    (output / "media-hash-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            hash_indexed_media(args.archive, args.index, args.previous_manifest, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
