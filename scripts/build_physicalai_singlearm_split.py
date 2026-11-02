"""Build an episode-grouped development split from pinned NVIDIA file metadata.

This script reads publisher metadata only. It does not download media, labels,
model weights, or access a reserved holdout payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import requests

DATASET_ID = "nvidia/PhysicalAI-Robotics-Manipulation-SingleArm"
DATASET_REVISION = "73c00c18e7ca108e91846294b20a1d7cd5de9340"
DATASET_SUBSET = "panda-open-cabinet-left"
WORLD_VIDEO_RE = re.compile(
    r"^panda-open-cabinet-left/videos/chunk-(\d{3})/"
    r"observation\.images\.world_camera/episode_(\d{6})\.mp4$"
)


def _rank_key(episode_index: int, seed: int) -> str:
    payload = (
        f"{DATASET_ID}@{DATASET_REVISION}|{DATASET_SUBSET}|seed={seed}|"
        f"episode_index={episode_index:06d}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_hash(rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        f"{row['episode_index']:06d}\t{row['video_sha256']}\n"
        for row in sorted(rows, key=lambda item: item["episode_index"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_split_manifest(
    siblings: Iterable[Mapping[str, Any]],
    *,
    total_episodes: int,
    seed: int = 17,
    preexposed_episode_indices: Iterable[int] = (),
) -> dict[str, Any]:
    """Return deterministic train/validation/reserved-holdout episode assignments."""
    if isinstance(total_episodes, bool) or total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    preexposed = set(preexposed_episode_indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in preexposed):
        raise ValueError("preexposed episode identities must be integers")

    by_episode: dict[int, dict[str, Any]] = {}
    for sibling in siblings:
        path = sibling.get("rfilename")
        if not isinstance(path, str):
            continue
        match = WORLD_VIDEO_RE.fullmatch(path)
        if match is None:
            continue
        chunk_index, episode_text = match.groups()
        episode_index = int(episode_text)
        if int(chunk_index) != episode_index // 1000:
            raise ValueError(f"episode chunk does not match identity: {path}")
        if episode_index in by_episode:
            raise ValueError(f"duplicate episode world-camera asset: {episode_index}")

        lfs = sibling.get("lfs")
        digest = lfs.get("sha256") if isinstance(lfs, Mapping) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest.lower())
        ):
            raise ValueError(f"world-camera asset needs a valid SHA-256: {path}")
        size = sibling.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"world-camera asset needs a positive size: {path}")

        by_episode[episode_index] = {
            "episode_index": episode_index,
            "video_path": path,
            "video_bytes": size,
            "video_sha256": digest.lower(),
            "split_key_sha256": _rank_key(episode_index, seed),
        }

    expected_ids = set(range(total_episodes))
    actual_ids = set(by_episode)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            "world-camera inventory must contain exactly one asset for every episode; "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    if not preexposed <= expected_ids:
        raise ValueError("preexposed episode identity is outside the source inventory")

    digests = [row["video_sha256"] for row in by_episode.values()]
    if len(digests) != len(set(digests)):
        raise ValueError("duplicate world-camera media hash across episode groups")

    ranked = sorted(by_episode.values(), key=lambda row: row["split_key_sha256"])
    train_count = (total_episodes * 70 + 50) // 100
    train_validation_end = (total_episodes * 85 + 50) // 100
    validation_end = max(train_count, train_validation_end)
    preexposed_rows = [row for row in ranked if row["episode_index"] in preexposed]
    unseen_rows = [row for row in ranked if row["episode_index"] not in preexposed]
    if len(preexposed_rows) > train_count:
        raise ValueError("preexposed episodes exceed the available train allocation")
    ordered_rows = preexposed_rows + unseen_rows
    for index, row in enumerate(ordered_rows):
        if index < train_count:
            row["split"] = "train"
        elif index < validation_end:
            row["split"] = "validation"
        else:
            row["split"] = "reserved_holdout"

    rows = sorted(ranked, key=lambda row: row["episode_index"])
    split_names = ("train", "validation", "reserved_holdout")
    split_rows = {
        name: [row for row in rows if row["split"] == name] for name in split_names
    }
    pilot_train = sorted(split_rows["train"], key=lambda row: row["split_key_sha256"])[
        : min(64, len(split_rows["train"]))
    ]
    pilot_validation = sorted(
        split_rows["validation"], key=lambda row: row["split_key_sha256"]
    )[: min(16, len(split_rows["validation"]))]
    return {
        "schema_version": 1,
        "candidate_id": "nvidia-physicalai-singlearm-cabinet-left",
        "source": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "subset": DATASET_SUBSET,
            "source_split": "train: episodes 0 through 1511",
            "source_license": "CC-BY-4.0",
        },
        "grouping": {
            "group_key": "episode_index",
            "group_scope": "all rows and camera/depth assets belonging to one episode",
            "scene_identity_status": "UNKNOWN; metadata has no separate scene/layout id",
            "exact_world_video_sha256_overlap": 0,
            "unique_world_videos": len(digests),
        },
        "split_policy": {
            "seed": seed,
            "rank": "sha256(dataset@revision|subset|seed|episode_index)",
            "allocation": (
                "70% train, 15% validation, remainder reserved_holdout; "
                "rounded to nearest group"
            ),
            "reserved_holdout_status": (
                "public source; episode-level action summary metadata was inspected; "
                "not a blind or sealed audit"
            ),
            "media_inputs_loaded": False,
            "labels_loaded_for_split": False,
            "preexposed_episode_indices_forced_to_train": sorted(preexposed),
        },
        "split_counts": {name: len(split_rows[name]) for name in split_names},
        "split_hashes": {name: _split_hash(split_rows[name]) for name in split_names},
        "pilot_selection": {
            "policy": "lowest split_key_sha256 within train and validation partitions",
            "train_episode_indices": [row["episode_index"] for row in pilot_train],
            "validation_episode_indices": [row["episode_index"] for row in pilot_validation],
            "reserved_holdout_accessed": False,
        },
        "episodes": rows,
    }


def fetch_pinned_siblings() -> list[dict[str, Any]]:
    url = (
        "https://huggingface.co/api/datasets/"
        f"{DATASET_ID}/revision/{DATASET_REVISION}?blobs=true"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    if data.get("sha") != DATASET_REVISION:
        raise ValueError("publisher API returned a different dataset revision")
    if data.get("cardData", {}).get("license") != "cc-by-4.0":
        raise ValueError("publisher API license metadata no longer matches the audited source")
    siblings = data.get("siblings")
    if not isinstance(siblings, list):
        raise ValueError("publisher API response omitted the file inventory")
    return siblings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--episodes", type=int, default=1512)
    args = parser.parse_args()

    manifest = build_split_manifest(
        fetch_pinned_siblings(),
        total_episodes=args.episodes,
        seed=args.seed,
        preexposed_episode_indices=(0, 1, 750, 1511),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": manifest["split_counts"]}))


if __name__ == "__main__":
    main()
