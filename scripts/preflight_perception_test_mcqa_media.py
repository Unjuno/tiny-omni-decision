"""Estimate exact Perception Test MCQA media needs from pinned ZIP directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

try:
    from scripts.acquire_perception_temporal_match_train_media import RangeFile
except ModuleNotFoundError:
    from acquire_perception_temporal_match_train_media import RangeFile

ARCHIVES = {
    "train": {
        "filename": "train_videos.zip",
        "url": (
            "https://storage.googleapis.com/dm-perception-test/zip_data/"
            "train_videos.zip?generation=1685446268139925"
        ),
        "size_bytes": 28_404_087_974,
        "etag": '"-CJXz5tD4nP8CEAE="',
        "generation": "1685446268139925",
    },
    "valid": {
        "filename": "valid_videos.zip",
        "url": (
            "https://storage.googleapis.com/dm-perception-test/zip_data/"
            "valid_videos.zip?generation=1685479088600941"
        ),
        "size_bytes": 75_375_143_328,
        "etag": '"-CO3u6PLynf8CEAE="',
        "generation": "1685479088600941",
    },
}
MAX_MEMBER_BYTES = 256_000_000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(corpus_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    for split, filename in (("train", "train.jsonl"), ("valid", "validation.jsonl")):
        manifest_split = "validation" if split == "valid" else split
        path = corpus_dir / filename
        if sha256_file(path) != manifest[manifest_split]["records_sha256"]:
            raise ValueError(f"frozen {split} corpus hash mismatch")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"{split} corpus contains another split")
        if len(rows) != manifest[manifest_split]["record_count"]:
            raise ValueError(f"{split} corpus count mismatch")
        result[split] = rows
    train_ids = {row["group_id"] for row in result["train"]}
    valid_ids = {row["group_id"] for row in result["valid"]}
    if train_ids & valid_ids:
        raise ValueError("train/validation video identity overlap")
    for split, rows in result.items():
        if len({row["source_video_id"] for row in rows}) != len(
            {row["group_id"] for row in rows}
        ):
            raise ValueError(f"{split} group and source video IDs differ")
    return result, manifest


def plan_split_members(
    split: str, video_ids: set[str], archive: zipfile.ZipFile
) -> dict[str, Any]:
    if split not in ARCHIVES:
        raise ValueError("only official train and validation video archives are supported")
    members: list[dict[str, Any]] = []
    for video_id in sorted(video_ids):
        member_name = f"videos/{video_id}.mp4"
        try:
            info = archive.getinfo(member_name)
        except KeyError as error:
            raise ValueError(f"media member is missing: {split}/{member_name}") from error
        if info.file_size <= 0 or info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"media member size is outside policy: {split}/{member_name}")
        if info.is_dir():
            raise ValueError(f"expected a video file: {split}/{member_name}")
        members.append(
            {
                "video_id": video_id,
                "split": split,
                "archive": ARCHIVES[split]["filename"],
                "member": member_name,
                "expanded_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
            }
        )
    return {
        "video_groups": len(video_ids),
        "expanded_bytes": sum(row["expanded_bytes"] for row in members),
        "compressed_bytes": sum(row["compressed_bytes"] for row in members),
        "members": members,
    }


def preflight(corpus_dir: Path, output_root: Path, min_free_bytes: int) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"new media output root must not exist: {output_root}")
    if output_root.drive.upper() != "D:":
        raise ValueError("Perception Test media output must stay on D:")
    splits, manifest = load_corpus(corpus_dir)
    if manifest["source"].get("official_test_opened") is not False:
        raise ValueError("frozen corpus does not attest official test isolation")

    plans: dict[str, dict[str, Any]] = {}
    archive_receipts: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid"):
        pin = ARCHIVES[split]
        remote = RangeFile(pin["url"], pin["size_bytes"], pin["etag"])
        with zipfile.ZipFile(remote) as archive:
            plans[split] = plan_split_members(
                split, {row["group_id"] for row in splits[split]}, archive
            )
        archive_receipts[split] = {
            "filename": pin["filename"],
            "url": pin["url"],
            "generation": pin["generation"],
            "size_bytes": pin["size_bytes"],
            "etag": pin["etag"],
            "directory_range_bytes_read": remote.bytes_read,
        }

    parent = output_root.parent
    while not parent.exists():
        parent = parent.parent
    free_bytes = shutil.disk_usage(parent).free
    required_bytes = sum(row["expanded_bytes"] for row in plans.values())
    if free_bytes < required_bytes + min_free_bytes:
        raise ValueError(
            f"insufficient D: capacity: free={free_bytes}, selected={required_bytes}, "
            f"required reserve={min_free_bytes}"
        )
    return {
        "status": "PASS_METADATA_ONLY_MEDIA_PREFLIGHT",
        "experiment_id": manifest["experiment_id"],
        "corpus_manifest_sha256": sha256_file(corpus_dir / "manifest.json"),
        "archive_receipts": archive_receipts,
        "splits": {
            split: {
                key: value
                for key, value in plan.items()
                if key != "members"
            }
            | {"members": plan["members"]}
            for split, plan in plans.items()
        },
        "total_video_groups": sum(plan["video_groups"] for plan in plans.values()),
        "total_expanded_bytes": required_bytes,
        "total_compressed_bytes": sum(
            plan["compressed_bytes"] for plan in plans.values()
        ),
        "d_free_bytes_before_acquisition": free_bytes,
        "minimum_d_free_bytes_after_acquisition": min_free_bytes,
        "media_payload_bytes_read": 0,
        "video_files_extracted": 0,
        "official_test_archive_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite media preflight report: {args.report}")
    min_free_bytes = int(args.min_free_gib * 1024**3)
    report = preflight(args.corpus_dir, args.output_root, min_free_bytes)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "splits"
            }
            | {
                "split_summary": {
                    split: {
                        key: value
                        for key, value in summary.items()
                        if key != "members"
                    }
                    for split, summary in report["splits"].items()
                },
                "report_sha256": sha256_file(args.report),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
