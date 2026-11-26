"""Audit exact overlap of canonically resized decoded video frames across splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image


def canonical_frame_digest(image: Image.Image, *, side: int = 32) -> str:
    if side < 1:
        raise ValueError("side must be positive")
    normalized = image.convert("RGB").resize(
        (side, side), resample=Image.Resampling.BILINEAR
    )
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def add_frame_digest(
    index: dict[str, dict[str, list[dict[str, int]]]],
    frame_hash: str,
    split: str,
    *,
    episode: int,
    frame_index: int,
) -> None:
    if split not in {"train", "validation"}:
        raise ValueError(f"unsupported split: {split}")
    index.setdefault(frame_hash, {"train": [], "validation": []})[split].append(
        {"episode": episode, "frame_index": frame_index}
    )


def summarize_frame_overlap(
    index: dict[str, dict[str, list[dict[str, int]]]],
    *,
    example_limit: int = 20,
    location_limit: int = 5,
) -> dict[str, Any]:
    if example_limit < 0 or location_limit < 1:
        raise ValueError("example_limit must be nonnegative and location_limit positive")
    matches = [
        frame_hash
        for frame_hash, split_rows in index.items()
        if split_rows["train"] and split_rows["validation"]
    ]
    matches.sort()
    examples = [
        {
            "frame_sha256": frame_hash,
            "train": index[frame_hash]["train"][:location_limit],
            "validation": index[frame_hash]["validation"][:location_limit],
        }
        for frame_hash in matches[:example_limit]
    ]
    return {
        "cross_split_unique_frame_hashes": len(matches),
        "cross_split_frame_occurrences": sum(
            len(index[frame_hash]["validation"]) for frame_hash in matches
        ),
        "examples": examples,
    }


def audit_receipt(
    receipt_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    side: int = 32,
    limit_per_split: int | None = None,
) -> dict[str, Any]:
    import av

    receipt_bytes = receipt_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    manifest = json.loads(manifest_bytes)
    if receipt["split_manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("receipt does not match the supplied split manifest")
    if receipt["revision"] != manifest["source"]["revision"]:
        raise ValueError("receipt and manifest revisions differ")
    if receipt["reserved_holdout_payload_downloaded"]:
        raise ValueError("receipt indicates reserved holdout payload was downloaded")
    if limit_per_split is not None and limit_per_split < 1:
        raise ValueError("limit_per_split must be positive")

    expected = {
        "train": manifest["pilot_selection"]["train_episode_indices"],
        "validation": manifest["pilot_selection"]["validation_episode_indices"],
    }
    if receipt["selection"]["train_episode_indices"] != expected["train"]:
        raise ValueError("receipt train ID order differs from manifest")
    if receipt["selection"]["validation_episode_indices"] != expected["validation"]:
        raise ValueError("receipt validation ID order differs from manifest")

    videos = [item for item in receipt["files"] if item["kind"] == "video"]
    selected: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    expected_sets = {split: set(ids) for split, ids in expected.items()}
    for item in videos:
        split = item["split"]
        if split not in selected or item["episode"] not in expected_sets[split]:
            raise ValueError("receipt contains video outside frozen pilot selection")
        selected[split].append(item)
    for split in selected:
        selected[split].sort(key=lambda item: expected[split].index(item["episode"]))
        if limit_per_split is not None:
            selected[split] = selected[split][:limit_per_split]
        if limit_per_split is None and [item["episode"] for item in selected[split]] != expected[
            split
        ]:
            raise ValueError(f"receipt does not contain the complete {split} pilot videos")

    frame_index: dict[str, dict[str, list[dict[str, int]]]] = {}
    frames_by_split = {"train": 0, "validation": 0}
    per_video = []
    started = time.perf_counter()
    processed_videos = 0
    for split in ("train", "validation"):
        for item in selected[split]:
            path = Path(item["local_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            sequence_hash = hashlib.sha256()
            frame_count = 0
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                for frame_count, frame in enumerate(container.decode(stream), start=1):
                    digest = canonical_frame_digest(frame.to_image(), side=side)
                    add_frame_digest(
                        frame_index,
                        digest,
                        split,
                        episode=item["episode"],
                        frame_index=frame_count - 1,
                    )
                    sequence_hash.update(bytes.fromhex(digest))
            if frame_count == 0:
                raise ValueError(f"video contains no decodable frames: {path}")
            frames_by_split[split] += frame_count
            per_video.append(
                {
                    "episode": item["episode"],
                    "split": split,
                    "video_sha256": item["sha256"],
                    "decoded_frame_count": frame_count,
                    "normalized_frame_sequence_sha256": sequence_hash.hexdigest(),
                }
            )
            processed_videos += 1
            print(
                json.dumps(
                    {
                        "processed_videos": processed_videos,
                        "episode": item["episode"],
                        "split": split,
                        "decoded_frames": frame_count,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )

    result = {
        "protocol": {
            "operation": "exact SHA-256 of decoded RGB frames after conversion and bilinear resize",
            "canonical_size": [side, side],
            "color_space": "RGB",
            "near_duplicate_similarity": "not tested",
            "reserved_holdout_accessed": False,
        },
        "source_revision": receipt["revision"],
        "subset": receipt["subset"],
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "limit_per_split": limit_per_split,
        "videos_by_split": {split: len(items) for split, items in selected.items()},
        "frames_by_split": frames_by_split,
        "unique_frame_hashes": len(frame_index),
        "overlap": summarize_frame_overlap(frame_index),
        "per_video": per_video,
        "wall_seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    result = audit_receipt(
        args.receipt,
        args.manifest,
        args.output,
        side=args.side,
        limit_per_split=args.limit_per_split,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "videos_by_split": result["videos_by_split"],
                "frames_by_split": result["frames_by_split"],
                "overlap": result["overlap"],
                "wall_seconds": result["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
