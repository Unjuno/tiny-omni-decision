"""Audit only the official AVA ActiveSpeaker TRAIN annotation archive.

This deliberately accepts one TRAIN archive only. It does not load validation,
challenge-test, or evaluation labels, and it never extracts files to disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_ARCHIVE_NAME = "ava_activespeaker_train_v1.0.tar.bz2"
MEMBER_PREFIX = "ava_activespeaker_train_v1.0/"
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]+$")
LABELS = {
    "SPEAKING_AUDIBLE",
    "SPEAKING_AND_AUDIBLE",
    "SPEAKING_NOT_AUDIBLE",
    "SPEAKING_BUT_NOT_AUDIBLE",
    "NOT_SPEAKING",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_train_archive(archive_path: Path) -> dict[str, object]:
    archive_path = archive_path.resolve(strict=True)
    if archive_path.name != EXPECTED_ARCHIVE_NAME:
        raise ValueError("only the official AVA ActiveSpeaker TRAIN archive is accepted")

    video_ids: set[str] = set()
    labels: Counter[str] = Counter()
    timestamps_by_video: dict[str, set[str]] = defaultdict(set)
    tracks_by_video: dict[str, set[str]] = defaultdict(set)
    row_count = 0
    invalid_counts: Counter[str] = Counter()
    invalid_samples: list[str] = []

    def record_invalid(location: str, reason: str) -> None:
        invalid_counts[reason] += 1
        if len(invalid_samples) < 25:
            invalid_samples.append(f"{location}:{reason}")

    with tarfile.open(archive_path, mode="r:bz2") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        if not members:
            raise ValueError("TRAIN archive contains no regular CSV members")

        for member in members:
            if not member.name.startswith(MEMBER_PREFIX) or ".." in Path(member.name).parts:
                raise ValueError(f"unexpected TRAIN member path: {member.name}")
            member_video_id = Path(member.name).name.removesuffix("-activespeaker.csv")
            if not member_video_id or not VIDEO_ID.fullmatch(member_video_id):
                raise ValueError(f"invalid video ID in TRAIN member: {member.name}")
            if member_video_id in video_ids:
                raise ValueError(f"duplicate TRAIN video member: {member_video_id}")
            video_ids.add(member_video_id)
            file_obj = archive.extractfile(member)
            if file_obj is None:
                raise ValueError(f"could not read regular TRAIN member: {member.name}")
            with io.TextIOWrapper(file_obj, encoding="utf-8", errors="strict", newline="") as text:
                for line_number, row in enumerate(csv.reader(text), 1):
                    location = f"{member.name}:{line_number}"
                    if len(row) != 8:
                        record_invalid(location, f"field_count={len(row)}")
                        continue
                    (
                        video_id,
                        timestamp_text,
                        x1_text,
                        y1_text,
                        x2_text,
                        y2_text,
                        label,
                        track_id,
                    ) = row
                    try:
                        timestamp = float(timestamp_text)
                        box = tuple(float(value) for value in (x1_text, y1_text, x2_text, y2_text))
                    except ValueError:
                        record_invalid(location, "invalid_numeric")
                        continue
                    if video_id != member_video_id:
                        record_invalid(location, "video_id_member_mismatch")
                    if (
                        timestamp < 0
                        or not math.isfinite(timestamp)
                        or not all(math.isfinite(value) for value in box)
                    ):
                        record_invalid(location, "nonfinite_or_negative")
                    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                        record_invalid(location, "invalid_normalized_box")
                    if label not in LABELS:
                        record_invalid(location, f"unknown_label={label}")
                    if not track_id:
                        record_invalid(location, "empty_track_id")
                    labels[label] += 1
                    timestamps_by_video[video_id].add(timestamp_text)
                    tracks_by_video[video_id].add(track_id)
                    row_count += 1
                    if row_count > 10_000_000:
                        raise ValueError("TRAIN annotation row safety limit exceeded")

    return {
        "dataset": "AVA ActiveSpeaker v1.0",
        "source_url": "https://research.google.com/ava/download/",
        "license": "CC-BY-4.0 (official AVA dataset page)",
        "archive_name": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "train_video_count": len(video_ids),
        "train_annotation_rows": row_count,
        "label_counts": dict(sorted(labels.items())),
        "per_video_timestamp_counts": {
            key: len(value) for key, value in sorted(timestamps_by_video.items())
        },
        "per_video_track_counts": {
            key: len(value) for key, value in sorted(tracks_by_video.items())
        },
        "invalid_row_counts": dict(sorted(invalid_counts.items())),
        "invalid_row_samples": invalid_samples,
        "validation_archive_opened": False,
        "challenge_test_archive_opened": False,
        "media_downloaded_or_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    result = audit_train_archive(args.train_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "train_video_count": result["train_video_count"],
                "train_annotation_rows": result["train_annotation_rows"],
                "label_counts": result["label_counts"],
                "invalid_row_counts": result["invalid_row_counts"],
                "validation_archive_opened": False,
                "challenge_test_archive_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
