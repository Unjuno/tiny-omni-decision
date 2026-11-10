"""Acquire exactly the frozen MCQA train/validation videos via pinned ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

try:
    from scripts.acquire_perception_temporal_match_train_media import (
        RangeFile,
        extract_member_bytes,
    )
except ModuleNotFoundError:
    from acquire_perception_temporal_match_train_media import RangeFile, extract_member_bytes

try:
    from scripts.preflight_perception_test_mcqa_media import ARCHIVES, load_corpus, sha256_file
except ModuleNotFoundError:
    from preflight_perception_test_mcqa_media import ARCHIVES, load_corpus, sha256_file


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def acquire_media(
    *,
    corpus_dir: Path,
    preflight_path: Path,
    expected_preflight_sha256: str,
    output_root: Path,
    min_free_bytes: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing media output: {output_root}")
    if output_root.drive.upper() != "D:":
        raise ValueError("Perception Test media output must stay on D:")
    preflight_sha = sha256_file(preflight_path)
    if preflight_sha != expected_preflight_sha256:
        raise ValueError("media preflight report hash mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    splits, corpus_manifest = load_corpus(corpus_dir)
    corpus_sha = sha256_file(corpus_dir / "manifest.json")
    if preflight.get("corpus_manifest_sha256") != corpus_sha:
        raise ValueError("preflight report belongs to a different frozen corpus")

    if preflight.get("official_test_archive_opened") is not False:
        raise ValueError("preflight does not attest official test isolation")
    total_expanded = int(preflight["total_expanded_bytes"])
    planned_total = sum(
        int(item["expanded_bytes"])
        for split in ("train", "valid")
        for item in preflight["splits"][split]["members"]
    )
    if total_expanded != planned_total:
        raise ValueError("preflight total does not match selected media member plan")
    parent = output_root.parent
    while not parent.exists():
        parent = parent.parent
    free_before = shutil.disk_usage(parent).free
    if free_before < total_expanded + min_free_bytes:
        raise ValueError("D: free space fell below media acquisition reserve")

    output_root.mkdir(parents=True, exist_ok=False)
    progress_path = output_root / "acquisition-progress.jsonl"
    started_at = time.time()
    files: list[dict[str, Any]] = []
    archive_reads: dict[str, int] = {}
    with progress_path.open("xb") as progress:
        for split in ("train", "valid"):
            pin = ARCHIVES[split]
            plan = preflight["splits"][split]["members"]
            expected_ids = {row["group_id"] for row in splits[split]}
            if {item["video_id"] for item in plan} != expected_ids:
                raise ValueError(f"{split} preflight member IDs differ from frozen corpus")
            remote = RangeFile(pin["url"], pin["size_bytes"], pin["etag"])
            split_dir = output_root / split
            split_dir.mkdir()
            with zipfile.ZipFile(remote) as archive:
                for selected in plan:
                    info = archive.getinfo(selected["member"])
                    if (
                        info.file_size != selected["expanded_bytes"]
                        or info.compress_size != selected["compressed_bytes"]
                        or f"{info.CRC:08x}" != selected["crc32"]
                    ):
                        raise ValueError(
                            f"ZIP member changed after preflight: {selected['member']}"
                        )
                    payload = extract_member_bytes(remote, info)
                    digest = _sha256_bytes(payload)
                    destination = split_dir / f"{selected['video_id']}.mp4"
                    with destination.open("xb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    record = {
                        **selected,
                        "local_relative_path": destination.relative_to(output_root).as_posix(),
                        "sha256": digest,
                        "crc32_verified": True,
                        "expanded_size_verified": True,
                    }
                    files.append(record)
                    progress.write(
                        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(
                            "utf-8"
                        )
                    )
                    progress.flush()
                    os.fsync(progress.fileno())
            archive_reads[split] = remote.bytes_read

    train_video_ids = {
        record["video_id"] for record in files if record["split"] == "train"
    }
    valid_video_ids = {
        record["video_id"] for record in files if record["split"] == "valid"
    }
    if train_video_ids & valid_video_ids:
        raise ValueError("cross-split video identity overlap in acquired media")
    if len(files) != sum(len(preflight["splits"][s]["members"]) for s in ("train", "valid")):
        raise ValueError("acquired media count differs from preflight")
    free_after = shutil.disk_usage(parent).free
    if free_after < min_free_bytes:
        raise RuntimeError("D: free-space reserve was crossed during media acquisition")

    manifest = {
        "schema_version": 1,
        "experiment_id": corpus_manifest["experiment_id"],
        "source": corpus_manifest["source"],
        "corpus_manifest_sha256": corpus_sha,
        "media_preflight_sha256": preflight_sha,
        "archive_receipts": {
            split: {
                **preflight["archive_receipts"][split],
                "member_range_bytes_read": archive_reads[split],
            }
            for split in ("train", "valid")
        },
        "train_video_id_overlap_with_validation": 0,
        "video_count": len(files),
        "expanded_bytes": sum(row["expanded_bytes"] for row in files),
        "compressed_bytes": sum(row["compressed_bytes"] for row in files),
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "started_at_epoch_seconds": started_at,
        "finished_at_epoch_seconds": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "official_test_archive_opened": False,
        "media_files": files,
    }
    manifest_path = output_root / "media-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["media_manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    manifest = acquire_media(
        corpus_dir=args.corpus_dir,
        preflight_path=args.preflight,
        expected_preflight_sha256=args.expected_preflight_sha256,
        output_root=args.output_root,
        min_free_bytes=int(args.min_free_gib * 1024**3),
    )
    print(
        json.dumps(
            {
                "status": "MEDIA_ACQUISITION_COMPLETE",
                "video_count": manifest["video_count"],
                "expanded_bytes": manifest["expanded_bytes"],
                "free_bytes_after": manifest["free_bytes_after"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "media_manifest_sha256": manifest["media_manifest_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
