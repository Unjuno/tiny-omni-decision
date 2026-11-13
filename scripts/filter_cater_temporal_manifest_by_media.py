"""Derive a CATER task manifest containing only intervals observable in media."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE_MANIFEST_SHA256 = {
    "train": "c8e6c911a3edefdc3bfb71114213011f243a9fe66ea7e1d772fbe2ff93dad217",
    "validation": "dd9e7920026228d1e8442aea45883c36f1def7f54e96dd126856165b20079fa0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def filter_by_media_frames(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    frame_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    available_frames = {}
    for record in frame_records:
        key = (str(record["split"]), str(record["video_id"]))
        if key in available_frames:
            raise ValueError(f"duplicate media frame-count record: {key}")
        count = int(record.get("decoded_frames") or record["stream_frames"])
        if count < 1:
            raise ValueError(f"invalid media frame count for {key}: {count}")
        available_frames[key] = count

    filtered: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    rejected = []
    split_by_video = {}
    input_ids = set()
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            if row.get("split") != split:
                raise ValueError(f"mislabelled {split} sample: {row.get('id')}")
            if row["id"] in input_ids:
                raise ValueError(f"duplicate task sample ID: {row['id']}")
            input_ids.add(row["id"])
            video_id = str(row["video_id"])
            previous = split_by_video.setdefault(video_id, split)
            if previous != split:
                raise ValueError(f"video occurs in both source splits: {video_id}")
            count = available_frames.get((split, video_id))
            if count is None:
                raise ValueError(f"missing frame-count record: {split}/{video_id}")
            end_frame = int(row["frame_interval"][1])
            if end_frame >= count:
                rejected.append(
                    {
                        "id": row["id"],
                        "split": split,
                        "video_id": video_id,
                        "end_frame": end_frame,
                        "available_frames": count,
                        "reason": "task endpoint is outside decoded media",
                    }
                )
            else:
                filtered[split].append(row)

    media_keys = set(available_frames)
    task_video_keys = {
        (split, str(row["video_id"]))
        for split, rows in (("train", train_rows), ("validation", validation_rows))
        for row in rows
    }
    if media_keys != task_video_keys:
        raise ValueError("frame-count inventory and task manifests contain different video IDs")
    train_videos = {row["video_id"] for row in filtered["train"]}
    validation_videos = {row["video_id"] for row in filtered["validation"]}
    if train_videos & validation_videos:
        raise ValueError("video overlap remains after media-frame filtering")

    summary = {
        "status": "PASS_MEDIA_ENDPOINT_FILTER",
        "filter_rule": (
            "keep a question only when its inclusive end frame is less than "
            "decoded frame count"
        ),
        "input_rows": {"train": len(train_rows), "validation": len(validation_rows)},
        "kept_rows": {split: len(rows) for split, rows in filtered.items()},
        "source_video_groups": {
            split: len({row["video_id"] for row in rows})
            for split, rows in (("train", train_rows), ("validation", validation_rows))
        },
        "kept_video_groups": {
            split: len({row["video_id"] for row in rows}) for split, rows in filtered.items()
        },
        "removed_rows_by_split": dict(sorted(Counter(row["split"] for row in rejected).items())),
        "option_count_unchanged": all(
            len(row["options"]) == int(row["num_options"])
            for rows in filtered.values()
            for row in rows
        ),
        "train_validation_video_overlap": 0,
        "removed_rows": rejected,
    }
    return filtered["train"], filtered["validation"], summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--frame-audit-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.manifest_dir.resolve(strict=True)
    frame_audit = args.frame_audit_jsonl.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite filtered manifest: {output}")
    for split in ("train", "validation"):
        actual = sha256_file(source / f"{split}.jsonl")
        if actual != SOURCE_MANIFEST_SHA256[split]:
            raise ValueError(f"unexpected source task manifest hash: {split}")
    train, validation, summary = filter_by_media_frames(
        read_jsonl(source / "train.jsonl"),
        read_jsonl(source / "validation.jsonl"),
        read_jsonl(frame_audit),
    )
    output.mkdir(parents=True)
    _write_jsonl(output / "train.jsonl", train)
    _write_jsonl(output / "validation.jsonl", validation)
    summary["input_manifest_sha256"] = {
        split: sha256_file(source / f"{split}.jsonl") for split in ("train", "validation")
    }
    summary["frame_audit_sha256"] = sha256_file(frame_audit)
    summary["output_manifest_sha256"] = {
        split: sha256_file(output / f"{split}.jsonl") for split in ("train", "validation")
    }
    summary["normalized_question_option_content_overlap"] = len(
        {
            (
                row["question"].casefold().strip(),
                tuple(sorted(option["id"] for option in row["options"])),
            )
            for row in train
        }
        & {
            (
                row["question"].casefold().strip(),
                tuple(sorted(option["id"] for option in row["options"])),
            )
            for row in validation
        }
    )
    (output / "filter-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
