"""Join the frozen synthetic AV protocol to its verified local video files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.create_clevrer_synthetic_av_protocol import DATASET, REVISION, sha256_bytes
from scripts.fetch_clevrer_synthetic_av_media import PROTOCOL_SHA256, _selected_scenes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_rows(
    protocol: dict[str, Any],
    split: str,
    media_by_scene: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for sample in protocol["samples"][split]:
        scene = sample.get("scene_index")
        if scene not in media_by_scene or sample.get("split") != split:
            raise ValueError(f"sample scene/split does not match frozen media receipt: {sample}")
        target_index = sample.get("target")
        options = sample.get("options")
        if (
            not isinstance(target_index, int)
            or isinstance(target_index, bool)
            or not isinstance(options, list)
            or len(options) != 2
            or not 0 <= target_index < len(options)
        ):
            raise ValueError("synthetic AV sample has invalid binary options or target")
        media = media_by_scene[scene]
        if sample.get("video_filename") != Path(media["path"]).name:
            raise ValueError(f"sample video filename differs from media receipt for scene {scene}")
        rows.append(
            {
                "id": sample["sample_id"],
                "sample_id": sample["sample_id"],
                "pair_id": sample["pair_id"],
                "event_id": sample["event_id"],
                "source": f"{DATASET}@{REVISION}",
                "split": split,
                "scene_index": scene,
                "scene_group_id": f"clevrer:{split}:{scene}",
                "media_path": PurePosixPath(media["path"]).as_posix(),
                "media_sha256": media["sha256"],
                "question": sample["question"],
                "options": list(options),
                "target": options[target_index],
                "target_index": target_index,
                "relation": sample["relation"],
                "collision_frame": sample["collision_frame"],
                "sound_onset_frame": sample["sound_onset_frame"],
                "sound_onset_sample": sample["sound_onset_sample"],
                "scene_time_shift_frames": sample["scene_time_shift_frames"],
                "video_frame_range": [sample["video_start_frame"], sample["video_end_frame"]],
                "object_ids": sample["object_ids"],
                "object_descriptors": sample["object_descriptors"],
            }
        )
    rows.sort(key=lambda row: row["id"])
    return rows


def _validate_pairs(rows: list[dict[str, Any]], split: str) -> None:
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate sample IDs in {split}")
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(row["pair_id"], []).append(row)
    for pair_id, pair in pairs.items():
        if (
            len(pair) != 2
            or {row["target_index"] for row in pair} != {0, 1}
            or len({row["scene_index"] for row in pair}) != 1
            or len({row["media_sha256"] for row in pair}) != 1
            or len({row["question"] for row in pair}) != 1
            or len({tuple(row["options"]) for row in pair}) != 1
        ):
            raise ValueError(f"incomplete or inconsistent paired AV variants: {pair_id}")


def materialize_samples(
    protocol_path: Path, receipt_path: Path, output_dir: Path
) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = sha256_bytes(protocol_bytes)
    if protocol_sha != PROTOCOL_SHA256:
        raise ValueError("protocol SHA-256 differs from frozen task protocol")
    protocol = json.loads(protocol_bytes)
    scenes = _selected_scenes(protocol)
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if (
        receipt.get("status") != "complete_train_only_media_fetch"
        or receipt.get("dataset") != DATASET
        or receipt.get("revision") != REVISION
        or receipt.get("protocol_sha256") != protocol_sha
        or receipt.get("validation_test_or_sealed_data_opened") is not False
    ):
        raise ValueError("media receipt does not match the frozen train-only protocol")
    entries = receipt.get("entries")
    if not isinstance(entries, list) or len(entries) != len(scenes):
        raise ValueError("media receipt entry count differs from frozen scene list")
    expected_train_scenes = set(protocol["split"]["scene_ids"]["train"])
    expected_validation_scenes = set(protocol["split"]["scene_ids"]["validation"])
    if (
        receipt.get("scene_overlap") != 0
        or receipt.get("media_sha256_overlap") != 0
        or receipt.get("train_scene_count") != len(expected_train_scenes)
        or receipt.get("validation_scene_count") != len(expected_validation_scenes)
    ):
        raise ValueError("media receipt split accounting differs from frozen protocol")
    media_by_scene = {}
    for entry in entries:
        scene = entry.get("scene_index")
        if scene in media_by_scene or scene not in scenes:
            raise ValueError(f"duplicate or unexpected scene in media receipt: {scene}")
        expected_split = (
            "train" if scene in protocol["split"]["scene_ids"]["train"] else "validation"
        )
        if entry.get("split") != expected_split:
            raise ValueError(f"media receipt split mismatch for scene {scene}")
        relative = PurePosixPath(entry.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe media relative path: {relative}")
        media_path = output_dir.joinpath(*relative.parts)
        if not media_path.is_file():
            raise FileNotFoundError(f"missing selected video: {media_path}")
        if media_path.stat().st_size != entry.get("bytes") or _sha256_file(media_path) != entry.get(
            "sha256"
        ):
            raise ValueError(f"media file hash mismatch for scene {scene}")
        media_by_scene[scene] = entry
    if set(media_by_scene) != set(scenes):
        raise ValueError("media receipt scene IDs differ from frozen protocol")
    if len({entry["sha256"] for entry in entries}) != len(entries):
        raise ValueError("duplicate media content hash across selected scenes")

    rows = {
        split: _sample_rows(protocol, split, media_by_scene) for split in ("train", "validation")
    }
    for split, split_rows in rows.items():
        _validate_pairs(split_rows, split)
    train_scenes = {row["scene_index"] for row in rows["train"]}
    validation_scenes = {row["scene_index"] for row in rows["validation"]}
    train_hashes = {row["media_sha256"] for row in rows["train"]}
    validation_hashes = {row["media_sha256"] for row in rows["validation"]}
    if train_scenes & validation_scenes or train_hashes & validation_hashes:
        raise ValueError("sample manifest train/validation asset overlap")

    output_files = [
        output_dir / "train.jsonl",
        output_dir / "validation.jsonl",
        output_dir / "fetch-report.json",
    ]
    if any(path.exists() for path in output_files):
        raise FileExistsError("sample manifests already exist; refusing to overwrite")
    for split in ("train", "validation"):
        path = output_dir / f"{split}.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows[split]:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "status": "complete_frozen_synthetic_av_sample_manifest",
        "dataset": DATASET,
        "revision": REVISION,
        "protocol_sha256": protocol_sha,
        "media_receipt_sha256": sha256_bytes(receipt_bytes),
        "train_samples": len(rows["train"]),
        "validation_samples": len(rows["validation"]),
        "train_scenes": len(train_scenes),
        "validation_scenes": len(validation_scenes),
        "train_sample_id_order_sha256": protocol["split"]["sample_id_order_sha256"]["train"],
        "validation_sample_id_order_sha256": protocol["split"]["sample_id_order_sha256"][
            "validation"
        ],
        "train_manifest_sha256": _sha256_file(output_files[0]),
        "validation_manifest_sha256": _sha256_file(output_files[1]),
        "scene_overlap": 0,
        "media_hash_overlap": 0,
        "validation_test_or_sealed_data_opened": False,
    }
    with output_files[2].open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--media-receipt", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    args = parser.parse_args()
    report = materialize_samples(args.protocol, args.media_receipt, args.media_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
