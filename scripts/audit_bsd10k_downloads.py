"""Verify downloaded BSD10k split membership, file hashes, and media identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.audit_bsd10k_metadata import ALLOWED_LICENSE_PATHS
except ModuleNotFoundError:  # direct `python scripts/...` execution
    from audit_bsd10k_metadata import ALLOWED_LICENSE_PATHS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_identity(path: Path) -> tuple[str, tuple[int, int, int, int]]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        rate = source.getframerate()
        width = source.getsampwidth()
        frames_count = source.getnframes()
        if (
            source.getcomptype() != "NONE"
            or channels != 1
            or rate != 44_100
            or width != 2
            or frames_count < 1
        ):
            raise ValueError(f"unexpected BSD10k PCM format: {path.name}")
        frames = source.readframes(frames_count)
    if len(frames) != frames_count * channels * width:
        raise ValueError(f"truncated WAV frames: {path.name}")
    digest = hashlib.sha256()
    digest.update(f"pcm-s16le-mono-{rate}:".encode("ascii"))
    digest.update(frames)
    return digest.hexdigest(), (rate, channels, width, frames_count)


def audit_downloaded_splits(
    split_manifest_path: Path,
    result_paths: list[Path],
    media_root: Path,
) -> dict[str, Any]:
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "tiny-omni-decision/bsd10k-group-split-v1":
        raise ValueError("unexpected BSD10k split manifest schema")
    rows = manifest.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("BSD10k split manifest has no examples")
    by_id: dict[str, dict[str, Any]] = {}
    group_splits: dict[str, str] = {}
    for row in rows:
        sound_id = str(row.get("sound_id", ""))
        group = str(row.get("uploader_group_sha256", ""))
        split = str(row.get("split", ""))
        if not sound_id or sound_id in by_id or split not in {"train", "validation", "test"}:
            raise ValueError("malformed or duplicate split manifest row")
        if row.get("license") not in ALLOWED_LICENSE_PATHS.values():
            raise ValueError("split manifest includes a non-allowlisted license")
        if len(group) != 64:
            raise ValueError("split manifest contains an invalid uploader group hash")
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError("uploader group overlaps splits")
        by_id[sound_id] = row

    results: dict[str, Path] = {}
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        split = str(result.get("split", ""))
        if split in results or split not in {"train", "validation", "test"}:
            raise ValueError("duplicate or invalid split result")
        if result.get("source_record_id") != manifest.get("source_record_id"):
            raise ValueError("download result source does not match split manifest")
        results[split] = result_path

    if not results:
        raise ValueError("at least one completed split result is required")

    split_details: dict[str, dict[str, Any]] = {}
    file_hash_splits: dict[str, set[str]] = defaultdict(set)
    pcm_hash_splits: dict[str, set[str]] = defaultdict(set)
    observed_id_splits: dict[str, set[str]] = defaultdict(set)
    observed_group_splits: dict[str, set[str]] = defaultdict(set)

    for split, result_path in sorted(results.items()):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected_ids = {sound_id for sound_id, row in by_id.items() if row["split"] == split}
        records = result.get("downloaded_files")
        if not isinstance(records, list) or len(records) != result.get("downloaded_count"):
            raise ValueError(f"download report count is inconsistent for {split}")
        actual_ids = [str(record.get("sound_id", "")) for record in records]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
            raise ValueError(f"downloaded IDs differ from the frozen {split} membership")

        bytes_total = 0
        pcm_formats: Counter[str] = Counter()
        for record in records:
            sound_id = str(record["sound_id"])
            expected = by_id[sound_id]
            if (
                record.get("license") != expected["license"]
                or record.get("class") != expected["class"]
            ):
                raise ValueError(
                    f"download record metadata differs from the split manifest: {split}"
                )
            path = media_root / split / f"{sound_id}.wav"
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"WAV byte count mismatch: {split}")
            digest = sha256_file(path)
            if digest != record.get("sha256"):
                raise ValueError(f"WAV SHA-256 mismatch: {split}")
            crc = 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    crc = zlib.crc32(block, crc)
            if f"{crc:08x}" != record.get("crc32"):
                raise ValueError(f"WAV CRC-32 mismatch: {split}")
            pcm_hash, audio_format = _media_identity(path)
            file_hash_splits[digest].add(split)
            pcm_hash_splits[pcm_hash].add(split)
            observed_id_splits[sound_id].add(split)
            observed_group_splits[expected["uploader_group_sha256"]].add(split)
            pcm_formats[str(audio_format)] += 1
            bytes_total += path.stat().st_size

        split_details[split] = {
            "example_count": len(records),
            "uploader_group_count": len(
                {by_id[sound_id]["uploader_group_sha256"] for sound_id in actual_ids}
            ),
            "uncompressed_media_bytes": bytes_total,
            "pcm_format_counts": dict(sorted(pcm_formats.items())),
        }

    def overlap(split_map: dict[str, set[str]]) -> int:
        return sum(len(splits) > 1 for splits in split_map.values())

    exact_media_overlap = overlap(file_hash_splits)
    normalized_pcm_overlap = overlap(pcm_hash_splits)
    id_overlap = overlap(observed_id_splits)
    group_overlap = overlap(observed_group_splits)
    return {
        "schema": "tiny-omni-decision/bsd10k-downloaded-media-audit-v1",
        "source_record_id": manifest["source_record_id"],
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "completed_splits": split_details,
        "cross_split_sound_id_overlap_count": id_overlap,
        "cross_split_uploader_group_overlap_count": group_overlap,
        "cross_split_file_sha256_overlap_count": exact_media_overlap,
        "cross_split_normalized_pcm_overlap_count": normalized_pcm_overlap,
        "claims_all_three_splits_complete": set(results) == {"train", "validation", "test"},
        "data_quality_status": (
            "PASS_NO_OVERLAP"
            if not any((id_overlap, group_overlap, exact_media_overlap, normalized_pcm_overlap))
            else "FAIL_CROSS_SPLIT_OVERLAP"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_downloaded_splits(args.split_manifest, args.result, args.media_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "completed_splits"}, indent=2))
    return 0 if report["data_quality_status"] == "PASS_NO_OVERLAP" else 2


if __name__ == "__main__":
    raise SystemExit(main())
