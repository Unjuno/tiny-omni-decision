"""Fetch only the frozen CLEVRER video cohort using verified ZIP byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

from scripts.fetch_clevrer_probe import REVISION, TRAIN_BYTES, TRAIN_ETAG, TRAIN_URL, RangeFile

DISK_RESERVE_BYTES = 8 * 2**30


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifests(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> None:
    if not train or not validation:
        raise ValueError("both corpus splits must be non-empty")
    for split, rows in (("train", train), ("validation", validation)):
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"all rows in {split} manifest must be marked {split}")
        if len({str(row.get("id")) for row in rows}) != len(rows):
            raise ValueError(f"duplicate sample IDs in {split}")
        if any(not str(row.get("video_filename", "")).endswith(".mp4") for row in rows):
            raise ValueError(f"invalid video filename in {split}")
    train_scenes = {int(row["scene_index"]) for row in train}
    validation_scenes = {int(row["scene_index"]) for row in validation}
    train_content = {
        (
            " ".join(str(row["question"]).casefold().split()),
            tuple(sorted(" ".join(str(option).casefold().split()) for option in row["options"])),
        )
        for row in train
    }
    validation_content = {
        (
            " ".join(str(row["question"]).casefold().split()),
            tuple(sorted(" ".join(str(option).casefold().split()) for option in row["options"])),
        )
        for row in validation
    }
    if train_scenes & validation_scenes:
        raise ValueError("scene identity overlaps across splits")
    if train_content & validation_content:
        raise ValueError("normalized question/options overlap across splits")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--disk-reserve-bytes", type=int, default=DISK_RESERVE_BYTES)
    args = parser.parse_args()
    corpus = args.corpus_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite media output directory: {output}")
    if args.disk_reserve_bytes < 0:
        raise ValueError("disk reserve must be non-negative")

    splits = {name: _read_jsonl(corpus / f"{name}.jsonl") for name in ("train", "validation")}
    validate_manifests(splits["train"], splits["validation"])
    unique_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for split, rows in splits.items():
        unique_by_split[split] = {}
        for row in rows:
            filename = str(row["video_filename"])
            previous = unique_by_split[split].setdefault(filename, row)
            if int(previous["scene_index"]) != int(row["scene_index"]):
                raise ValueError(f"video filename maps to multiple scenes: {filename}")
    free = shutil.disk_usage(output.parent).free

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    archive_bytes_read: dict[str, int] = {}
    archive_entries = 0
    total_media_bytes = 0
    try:
        for split, rows in splits.items():
            archive_file = RangeFile(TRAIN_URL, TRAIN_BYTES, TRAIN_ETAG)
            with zipfile.ZipFile(archive_file) as archive:
                archive_entries += len(archive.infolist())
                names_by_basename: dict[str, list[zipfile.ZipInfo]] = {}
                for info in archive.infolist():
                    if info.filename.lower().endswith(".mp4"):
                        names_by_basename.setdefault(Path(info.filename).name, []).append(info)
                selected: dict[str, zipfile.ZipInfo] = {}
                for filename in unique_by_split[split]:
                    matches = names_by_basename.get(filename, [])
                    if len(matches) != 1:
                        raise ValueError(
                            "expected exactly one archive member for "
                            f"{filename}; got {len(matches)}"
                        )
                    selected[filename] = matches[0]
                estimate = sum(info.file_size for info in selected.values())
                total_media_bytes += estimate
                if free - total_media_bytes < args.disk_reserve_bytes:
                    raise OSError(
                        "insufficient disk space for selected media plus reserve: "
                        f"free={free}, selected_estimate={total_media_bytes}, "
                        f"reserve={args.disk_reserve_bytes}"
                    )
                extracted: dict[str, tuple[str, int, str]] = {}
                media_dir = output / "media" / split
                media_dir.mkdir(parents=True, exist_ok=True)
                for filename, info in selected.items():
                    media = archive.read(info)
                    if len(media) != info.file_size or len(media) < 16 or b"ftyp" not in media[:16]:
                        raise ValueError(f"invalid MP4 bytes for {info.filename}")
                    media_hash = sha256_bytes(media)
                    (media_dir / filename).write_bytes(media)
                    extracted[filename] = (info.filename, len(media), media_hash)
                archive_bytes_read[split] = archive_file.bytes_read
            for row in rows:
                member, media_bytes, media_hash = extracted[str(row["video_filename"])]
                row.update(
                    {
                        "source": f"MIT-IBM/CLEVRER@{REVISION}",
                        "media_path": f"media/{split}/{row['video_filename']}",
                        "media_bytes": media_bytes,
                        "media_sha256": media_hash,
                        "archive_member": member,
                    }
                )
        train_hashes = {row["media_sha256"] for row in splits["train"]}
        validation_hashes = {row["media_sha256"] for row in splits["validation"]}
        if train_hashes & validation_hashes:
            raise ValueError("media SHA-256 overlap across train and validation")
        for split, rows in splits.items():
            with (output / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report = {
            "status": "complete_verified_clevrer_temporal_media",
            "dataset_revision": REVISION,
            "corpus_dir": str(corpus),
            "corpus_manifest_sha256": {
                f"{split}.jsonl": sha256_file(corpus / f"{split}.jsonl") for split in splits
            },
            "train_rows": len(splits["train"]),
            "validation_rows": len(splits["validation"]),
            "train_unique_videos": len(unique_by_split["train"]),
            "validation_unique_videos": len(unique_by_split["validation"]),
            "train_scene_overlap": 0,
            "media_sha256_overlap": 0,
            "archive_url": TRAIN_URL,
            "archive_size_bytes": TRAIN_BYTES,
            "archive_etag": TRAIN_ETAG,
            "archive_sha256_verified": False,
            "archive_directory_entries_seen_per_split": archive_entries,
            "archive_range_bytes_read": archive_bytes_read,
            "media_bytes_uncompressed": total_media_bytes,
            "disk_free_before_bytes": free,
            "disk_reserve_bytes": args.disk_reserve_bytes,
            "media_files_verified": sum(len(files) for files in unique_by_split.values()),
            "media_hash_manifest_sha256": {
                split: sha256_bytes(
                    "\n".join(sorted({row["media_sha256"] for row in rows})).encode("ascii")
                )
                for split, rows in splits.items()
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        (output / "fetch-report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        # Preserve partial bytes and manifests for diagnosis; never reuse this output path.
        raise
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "train_videos": report["train_unique_videos"],
                "validation_videos": report["validation_unique_videos"],
                "media_bytes": report["media_bytes_uncompressed"],
                "archive_range_bytes": report["archive_range_bytes_read"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
