"""Freeze an episode-grouped split from pinned NVIDIA stack-platforms metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DATASET_ID = "nvidia/PhysicalAI-Robotics-Manipulation-SingleArm"
DATASET_REVISION = "73c00c18e7ca108e91846294b20a1d7cd5de9340"
DATASET_SUBSET = "panda-stack-platforms"
TOTAL_EPISODES = 17_629
INFO_SHA256 = "325ee4a7354291b57714a6980a8aecbc7f9a01d5ea66caf569cc82c8ad39c87a"
EPISODES_SHA256 = "d3d3a9f88df886d2fb005dbe4733d25020d1b0ac296046ea37eb59dda2752c73"
WORLD_VIDEO_RE = re.compile(
    r"^panda-stack-platforms/videos/chunk-(\d{3})/"
    r"observation\.images\.world_camera/episode_(\d{6})\.mp4$"
)
EPISODE_DATA_RE = re.compile(
    r"^panda-stack-platforms/data/chunk-(\d{3})/episode_(\d{6})\.parquet$"
)


def _rank_key(episode_index: int, seed: int) -> str:
    text = (
        f"{DATASET_ID}@{DATASET_REVISION}|{DATASET_SUBSET}|seed={seed}|"
        f"episode_index={episode_index:06d}"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_hash(rows: list[dict[str, Any]]) -> str:
    text = "".join(
        f"{row['episode_index']:06d}\t{row['video_sha256']}\t{row['parquet_sha256']}\n"
        for row in sorted(rows, key=lambda row: row["episode_index"])
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _asset_inventory_hash(rows: list[dict[str, Any]], *, kind: str) -> str:
    text = "".join(
        f"{row[f'{kind}_path']}\t{row[f'{kind}_bytes']}\t{row[f'{kind}_sha256']}\n"
        for row in sorted(rows, key=lambda row: row["episode_index"])
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _id_order_hash(rows: list[dict[str, Any]]) -> str:
    text = "\n".join(f"{row['episode_index']:06d}" for row in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(
    world_videos: Iterable[Mapping[str, Any]],
    episode_data: Iterable[Mapping[str, Any]],
    *,
    total_episodes: int = TOTAL_EPISODES,
    seed: int = 17,
    pilot_train_groups: int = 700,
    pilot_validation_groups: int = 250,
    full_subset_bytes: int | None = None,
) -> dict[str, Any]:
    """Build deterministic train/validation/reserved splits without media access."""
    if isinstance(total_episodes, bool) or total_episodes < 3:
        raise ValueError("total_episodes must be at least three")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if pilot_train_groups < 1 or pilot_validation_groups < 1:
        raise ValueError("pilot selections must be positive")

    world_items = list(world_videos)
    data_items = list(episode_data)

    def parse_asset(
        items: Iterable[Mapping[str, Any]], pattern: re.Pattern[str], kind: str
    ) -> dict[int, dict[str, Any]]:
        by_episode: dict[int, dict[str, Any]] = {}
        for item in items:
            path = item.get("path") or item.get("rfilename")
            if not isinstance(path, str):
                continue
            match = pattern.fullmatch(path)
            if match is None:
                continue
            chunk, episode_text = match.groups()
            episode = int(episode_text)
            if int(chunk) != episode // 1000:
                raise ValueError(f"episode chunk does not match identity: {path}")
            if episode in by_episode:
                raise ValueError(f"duplicate {kind} asset for episode {episode}")
            size = item.get("size")
            digest = item.get("sha256")
            lfs = item.get("lfs")
            if digest is None and isinstance(lfs, Mapping):
                digest = lfs.get("sha256")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError(f"{kind} asset requires positive size: {path}")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.lower())
            ):
                raise ValueError(f"{kind} asset requires LFS SHA-256: {path}")
            by_episode[episode] = {
                f"{kind}_path": path,
                f"{kind}_bytes": size,
                f"{kind}_sha256": digest.lower(),
            }
        return by_episode

    world_by_episode = parse_asset(world_items, WORLD_VIDEO_RE, "video")
    data_by_episode = parse_asset(data_items, EPISODE_DATA_RE, "parquet")
    by_episode: dict[int, dict[str, Any]] = {}
    for episode, video in world_by_episode.items():
        data = data_by_episode.get(episode)
        if data is None:
            continue
        by_episode[episode] = {
            "episode_index": episode,
            **video,
            **data,
            "split_key_sha256": _rank_key(episode, seed),
        }
    for item in world_items:
        path = item.get("path") or item.get("rfilename")
        if isinstance(path, str) and WORLD_VIDEO_RE.fullmatch(path):
            video_episode = int(WORLD_VIDEO_RE.fullmatch(path).group(2))
            if video_episode not in data_by_episode:
                raise ValueError(f"missing episode Parquet asset for {video_episode}")
    for item in data_items:
        path = item.get("path") or item.get("rfilename")
        if isinstance(path, str) and EPISODE_DATA_RE.fullmatch(path):
            data_episode = int(EPISODE_DATA_RE.fullmatch(path).group(2))
            if data_episode not in world_by_episode:
                raise ValueError(f"missing world-camera asset for {data_episode}")

    expected = set(range(total_episodes))
    if set(by_episode) != expected:
        raise ValueError(
            "world-camera inventory must contain exactly one file for every episode; "
            f"found={len(by_episode)} expected={total_episodes}"
        )
    video_digests = [row["video_sha256"] for row in by_episode.values()]
    if len(video_digests) != len(set(video_digests)):
        raise ValueError("duplicate world-camera SHA-256 across episode groups")
    parquet_digests = [row["parquet_sha256"] for row in by_episode.values()]
    if len(parquet_digests) != len(set(parquet_digests)):
        raise ValueError("duplicate episode-Parquet SHA-256 across episode groups")

    ranked = sorted(by_episode.values(), key=lambda row: row["split_key_sha256"])
    train_end = (total_episodes * 70 + 50) // 100
    validation_end = (total_episodes * 85 + 50) // 100
    for index, row in enumerate(ranked):
        row["split"] = (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "reserved_holdout"
        )
    rows = sorted(ranked, key=lambda row: row["episode_index"])
    grouped = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "reserved_holdout")
    }
    pilot_train = sorted(grouped["train"], key=lambda row: row["split_key_sha256"])[
        :pilot_train_groups
    ]
    pilot_validation = sorted(
        grouped["validation"], key=lambda row: row["split_key_sha256"]
    )[:pilot_validation_groups]
    if len(pilot_train) != pilot_train_groups:
        raise ValueError("pilot train selection exceeds available train groups")
    if len(pilot_validation) != pilot_validation_groups:
        raise ValueError("pilot validation selection exceeds available validation groups")

    return {
        "schema_version": 1,
        "candidate_id": "nvidia-physicalai-singlearm-stack-platforms",
        "status": "REVIEW_NOT_ADMITTED",
        "source": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "subset": DATASET_SUBSET,
            "publisher_license": "CC-BY-4.0",
            "publisher_info_json_sha256": INFO_SHA256,
            "publisher_episodes_jsonl_sha256": EPISODES_SHA256,
            "publisher_source_split": "train: episodes 0 through 17628",
        },
        "grouping": {
            "group_key": "episode_index",
            "all_rows_and_media_inherit_episode_group": True,
            "scene_or_layout_identity_status": "UNKNOWN; not present in pinned episode metadata",
            "unique_world_video_sha256": len(set(video_digests)),
            "unique_episode_parquet_sha256": len(set(parquet_digests)),
            "cross_split_exact_world_video_sha256_overlap": 0,
            "cross_split_exact_episode_parquet_sha256_overlap": 0,
        },
        "asset_inventory_hashes": {
            "world_video_sha256": _asset_inventory_hash(rows, kind="video"),
            "episode_parquet_sha256": _asset_inventory_hash(rows, kind="parquet"),
        },
        "split_policy": {
            "seed": seed,
            "rank": "sha256(dataset@revision|subset|seed|episode_index)",
            "allocation": "70% train, 15% validation, remainder reserved_holdout",
            "reserved_payload_loaded": False,
        },
        "split_counts": {name: len(items) for name, items in grouped.items()},
        "split_hashes": {name: _split_hash(items) for name, items in grouped.items()},
        "full_subset_publisher_bytes": full_subset_bytes,
        "world_video_inventory_bytes": sum(row["video_bytes"] for row in rows),
        "episode_parquet_inventory_bytes": sum(row["parquet_bytes"] for row in rows),
        "pilot_selection": {
            "selection_rank": "split_key_sha256 ascending within assigned split",
            "train_episode_indices": [row["episode_index"] for row in pilot_train],
            "validation_episode_indices": [row["episode_index"] for row in pilot_validation],
            "train_group_count": len(pilot_train),
            "validation_group_count": len(pilot_validation),
            "train_episode_id_order_sha256": _id_order_hash(pilot_train),
            "validation_episode_id_order_sha256": _id_order_hash(pilot_validation),
            "train_video_bytes": sum(row["video_bytes"] for row in pilot_train),
            "validation_video_bytes": sum(row["video_bytes"] for row in pilot_validation),
            "train_parquet_bytes": sum(row["parquet_bytes"] for row in pilot_train),
            "validation_parquet_bytes": sum(row["parquet_bytes"] for row in pilot_validation),
            "selected_media_payload_bytes": sum(
                row["video_bytes"] + row["parquet_bytes"]
                for row in pilot_train + pilot_validation
            ),
        },
        "episode_assignments": rows,
    }


def inventory_metadata() -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Read pinned Hub file metadata only; no media bytes are requested."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    info = api.dataset_info(DATASET_ID, revision=DATASET_REVISION, token=False)
    if info.sha != DATASET_REVISION:
        raise ValueError("Hub returned a different dataset revision")
    files = api.list_repo_tree(
        DATASET_ID,
        path_in_repo=DATASET_SUBSET,
        recursive=True,
        expand=False,
        revision=DATASET_REVISION,
        repo_type="dataset",
        token=False,
    )
    world = []
    episode_data = []
    subset_bytes = 0
    for item in files:
        if not isinstance(item, RepoFile):
            continue
        subset_bytes += item.size or 0
        if "/videos/" not in item.path or "/observation.images.world_camera/" not in item.path:
            if EPISODE_DATA_RE.fullmatch(item.path):
                lfs = getattr(item, "lfs", None)
                episode_data.append(
                    {
                        "path": item.path,
                        "size": item.size,
                        "sha256": getattr(lfs, "sha256", None),
                    }
                )
            continue
        lfs = getattr(item, "lfs", None)
        world.append({"path": item.path, "size": item.size, "sha256": getattr(lfs, "sha256", None)})
    return world, episode_data, subset_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-train-groups", type=int, default=700)
    parser.add_argument("--pilot-validation-groups", type=int, default=250)
    args = parser.parse_args()
    world, episode_data, subset_bytes = inventory_metadata()
    manifest = build_manifest(
        world,
        episode_data,
        pilot_train_groups=args.pilot_train_groups,
        pilot_validation_groups=args.pilot_validation_groups,
        full_subset_bytes=subset_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "subset_bytes": subset_bytes,
                "split_counts": manifest["split_counts"],
                "pilot_selection": {
                    key: value
                    for key, value in manifest["pilot_selection"].items()
                    if not key.endswith("episode_indices")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
