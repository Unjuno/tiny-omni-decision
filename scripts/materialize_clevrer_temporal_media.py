"""Fetch only the pinned CLEVRER videos referenced by a frozen temporal split."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.fetch_clevrer_probe import REVISION, TRAIN_BYTES, TRAIN_ETAG, TRAIN_URL, RangeFile


def _stream_member_to_file(archive: Any, member: str, target: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(member) as source, target.open("xb") as destination:
        first = source.read(16)
        if len(first) < 16 or b"ftyp" not in first:
            raise ValueError(f"invalid MP4 media bytes for {member}")
        destination.write(first)
        digest.update(first)
        size += len(first)
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _normalized_content(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    question = " ".join(str(row["question"]).casefold().split())
    options = tuple(sorted(" ".join(str(item).casefold().split()) for item in row["options"]))
    return question, options


def materialize_media(
    rows_by_split: dict[str, list[dict[str, Any]]],
    archive: Any,
    output_dir: Path,
    *,
    revision: str,
    archive_opener: Callable[[], Any] | None = None,
    worker_count: int = 1,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if set(rows_by_split) != {"train", "validation"}:
        raise ValueError("rows_by_split must contain exactly train and validation")
    scene_ids: dict[str, set[int]] = {}
    for split, rows in rows_by_split.items():
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"{split} row has a mismatched split label")
        scene_ids[split] = {int(row["scene_index"]) for row in rows}
    if scene_ids["train"] & scene_ids["validation"]:
        raise ValueError("CLEVRER scene overlap across train and validation")
    train_content = {_normalized_content(row) for row in rows_by_split["train"]}
    validation_content = {_normalized_content(row) for row in rows_by_split["validation"]}
    if train_content & validation_content:
        raise ValueError("normalized question/options overlap across train and validation")
    if worker_count < 1 or (worker_count > 1 and archive_opener is None):
        raise ValueError("parallel media extraction requires a positive worker count and opener")

    names_by_basename: dict[str, list[str]] = {}
    for member in archive.namelist():
        if member.casefold().endswith(".mp4"):
            names_by_basename.setdefault(Path(member).name, []).append(member)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    scene_media: dict[str, dict[int, tuple[str, int, str]]] = {"train": {}, "validation": {}}
    media_hashes: dict[str, set[str]] = {"train": set(), "validation": set()}
    unique_assets: dict[tuple[str, int], tuple[str, str]] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            identity = (split, int(row["scene_index"]))
            filename = str(row["video_filename"])
            previous = unique_assets.setdefault(identity, (filename, ""))
            if previous[0] != filename:
                raise ValueError(f"scene {identity[1]} maps to multiple filenames")
    asset_tasks = []
    for (split, scene), (filename, _) in sorted(unique_assets.items()):
        matches = names_by_basename.get(filename, [])
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one archive member for {filename}, found {len(matches)}"
            )
        asset_tasks.append((split, scene, filename, matches[0]))

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent
    ) as tmp:
        staging = Path(tmp)
        archives: list[Any] = [archive]
        with contextlib.ExitStack() as stack:
            if worker_count > 1:
                archives.extend(
                    stack.enter_context(archive_opener()) for _ in range(worker_count - 1)
                )
                for worker_archive in archives[1:]:
                    worker_archive.namelist()
            batches = [asset_tasks[index::worker_count] for index in range(worker_count)]

            def extract_batch(index: int):
                worker_archive = archives[index]
                extracted = {}
                for split, scene, filename, member in batches[index]:
                    target = staging / "media" / split / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    size, digest = _stream_member_to_file(worker_archive, member, target)
                    extracted[(split, scene)] = (member, size, digest)
                byte_count = getattr(
                    getattr(worker_archive, "fp", None), "bytes_read", None
                )
                return extracted, byte_count

            if worker_count == 1:
                extraction_results = [extract_batch(0)]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
                    futures = [pool.submit(extract_batch, index) for index in range(worker_count)]
                    extraction_results = [future.result() for future in futures]
        archive_bytes = [result[1] for result in extraction_results if result[1] is not None]
        archive_bytes_read = sum(archive_bytes) if archive_bytes else None
        for extracted, _ in extraction_results:
            for (split, scene), metadata in extracted.items():
                scene_media[split][scene] = metadata
                media_hashes[split].add(metadata[2])

        for split, rows in rows_by_split.items():
            for row in rows:
                scene = int(row["scene_index"])
                metadata = scene_media[split][scene]
                member, media_bytes, media_hash = metadata
                row.update(
                    {
                        "source": f"MIT-IBM/CLEVRER@{revision}",
                        "scene_group_id": f"clevrer:{split}:{scene}",
                        "media_path": f"media/{split}/{row['video_filename']}",
                        "media_bytes": media_bytes,
                        "media_sha256": media_hash,
                        "archive_member": member,
                    }
                )
            with (staging / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        if media_hashes["train"] & media_hashes["validation"]:
            raise ValueError("CLEVRER media content overlap across train and validation")
        report = {
            "status": "complete_temporal_media_sample",
            "dataset": "MIT-IBM/CLEVRER",
            "revision": revision,
            "train_examples": len(rows_by_split["train"]),
            "validation_examples": len(rows_by_split["validation"]),
            "train_scene_count": len(scene_ids["train"]),
            "validation_scene_count": len(scene_ids["validation"]),
            "scene_overlap": 0,
            "media_sha_overlap": 0,
            "normalized_question_overlap": 0,
            "train_media_count": len(media_hashes["train"]),
            "validation_media_count": len(media_hashes["validation"]),
            "media_hash_overlap": 0,
            "media_bytes": {
                split: sum(size for _, size, _ in scene_media[split].values())
                for split in ("train", "validation")
            },
            "worker_count": worker_count,
            "range_archive_bytes_read": archive_bytes_read,
            "archive_sha256_verified": False,
            "limitations": [
                "selected video hashes are verified; the full video archive SHA-256 is not computed"
            ],
        }
        (staging / "fetch-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        gate = {
            "status": "pretraining_gate_pass",
            "scene_overlap": 0,
            "media_sha_overlap": 0,
            "normalized_question_overlap": 0,
            "archive_sha256_verified": False,
            "dataset_revision": revision,
            "selection_uses_sealed_audit": False,
        }
        (staging / "pretraining-gate.json").write_text(
            json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    return report


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    preflight = json.loads((args.sample_dir / "preflight.json").read_text(encoding="utf-8"))
    rows = {
        split: _read_rows(args.sample_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    for split in rows:
        expected = preflight["counts"].get(f"{split}.jsonl_sha256")
        if expected is None:
            expected = preflight["counts"]["split_file_sha256"][f"{split}.jsonl"]
        actual = hashlib.sha256((args.sample_dir / f"{split}.jsonl").read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen {split} split hash does not match preflight manifest")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    def open_archive():
        archive_file = RangeFile(TRAIN_URL, TRAIN_BYTES, TRAIN_ETAG)
        return zipfile.ZipFile(archive_file)

    with open_archive() as archive:
        report = materialize_media(
            rows,
            archive,
            args.output_dir,
            revision=REVISION,
            archive_opener=open_archive,
            worker_count=args.workers,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
