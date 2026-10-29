"""Freeze a small group-disjoint EgoViz video/IMU correspondence pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

INDEX_SHA256 = "c83126637cc24a7ac2b3600724b50cf9c53966b9b098acc57d5534e14ade1074"
SPLIT_SHA256 = "d2e2869dcce0a7d96cddc7dee639794a7cf253c3f922637ab9c9699cb5e6d7d2"
SPLIT_PREFIX = "seed17:egoviz-session-split-v1:"
CATEGORIES = [
    "cleaning_restroom",
    "organizing_items",
    "preparing_food",
    "mopping_floor",
]
ATTEMPT_01 = Path(r"C:\CodexArtifacts\egoviz-mcap-preflight-20261029-attempt-01")
ATTEMPT_02 = Path(r"C:\CodexArtifacts\egoviz-mcap-preflight-20261029-attempt-02")
EXISTING_FILES = {
    chunk_id: str(directory / f"{chunk_id}.mcap")
    for directory, chunk_id in (
        (ATTEMPT_01, "c59801fe4172cf9b430cd9232dda3909"),
        (ATTEMPT_01, "ad15950002d490a519a26e18052fc1e6"),
        (ATTEMPT_02, "7f2a80d98e54f0559893a5bf5cee0e5c"),
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_split(index: pd.DataFrame, split: dict[str, Any]) -> tuple[set[str], set[str]]:
    groups = sorted(
        index["group_id"].astype(str).unique(),
        key=lambda group: hashlib.sha256((SPLIT_PREFIX + group).encode()).hexdigest(),
    )
    development = set(groups[: round(len(groups) * 0.2)])
    training = set(groups) - development
    if {hashlib.sha256(group.encode()).hexdigest() for group in development} != set(
        split["development_group_sha256"]
    ):
        raise ValueError("development groups differ from the frozen split receipt")
    if {hashlib.sha256(group.encode()).hexdigest() for group in training} != set(
        split["train_group_sha256"]
    ):
        raise ValueError("train groups differ from the frozen split receipt")
    return training, development


def select(index_path: Path, split_path: Path) -> dict[str, Any]:
    if _sha256(index_path) != INDEX_SHA256 or _sha256(split_path) != SPLIT_SHA256:
        raise ValueError("source index or split receipt SHA-256 changed")
    index = pd.read_parquet(index_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("seed") != 17 or split.get("index_sha256") != INDEX_SHA256:
        raise ValueError("unexpected seed or source index in split receipt")
    training, development = _verify_split(index, split)

    selected: list[dict[str, Any]] = []
    split_groups: dict[str, set[str]] = {"train": set(), "development": set()}
    for split_name, allowed_groups in (("train", training), ("development", development)):
        used_groups: set[str] = set()
        for category in CATEGORIES:
            candidates = index[
                (index["category"] == category) & index["group_id"].astype(str).isin(allowed_groups)
            ].sort_values(["mcap_bytes", "chunk_id"])
            candidates = candidates.drop_duplicates("group_id")
            added = 0
            for row in candidates.to_dict(orient="records"):
                group_id = str(row["group_id"])
                if group_id in used_groups:
                    continue
                used_groups.add(group_id)
                split_groups[split_name].add(group_id)
                selected.append(
                    {
                        "split": split_name,
                        "category": category,
                        "group_id": group_id,
                        "sequence_id": str(row["sequence_id"]),
                        "chunk_id": str(row["chunk_id"]),
                        "mcap_path": str(row["mcap_path"]),
                        "expected_bytes": int(row["mcap_bytes"]),
                        "duration_s": float(row["duration_s"]),
                        "n_frames": int(row["n_frames"]),
                        "existing_source_path": EXISTING_FILES.get(str(row["chunk_id"])),
                    }
                )
                added += 1
                if added == 2:
                    break
            if added != 2:
                raise ValueError(f"fewer than two unique {split_name} groups for {category}")

    if split_groups["train"] & split_groups["development"]:
        raise ValueError("selected train and development groups overlap")
    if len(split_groups["train"]) != 8 or len(split_groups["development"]) != 8:
        raise ValueError("expected eight unique groups per split")
    total_bytes = sum(row["expected_bytes"] for row in selected)
    return {
        "format": "egoviz-vs-sync-selection-v1",
        "status": "PREREGISTERED_SOURCE_INPUT_PILOT_NOT_PRODUCT_EVALUATION",
        "source": {
            "dataset": "humaidtech/EgoViz-120",
            "index_sha256": INDEX_SHA256,
            "split_receipt_sha256": SPLIT_SHA256,
            "split_seed": 17,
            "split_unit": "publisher group_id session; participant identity is not verified",
            "bucket_revision": (
                "mutable; individual local files will be pinned by SHA-256 after retrieval"
            ),
        },
        "selection": {
            "category_order": CATEGORIES,
            "rule": (
                "For each split/category, rank records by (mcap_bytes, chunk_id), "
                "retain the first record per group_id, skip groups already selected in that split, "
                "and take the first two. No labels or media content are used for selection."
            ),
            "train_groups": sorted(split_groups["train"]),
            "development_groups": sorted(split_groups["development"]),
            "train_development_group_overlap": 0,
            "unique_groups_per_split": 8,
            "files_total": len(selected),
            "expected_total_bytes": total_bytes,
            "expected_new_download_bytes": sum(
                row["expected_bytes"] for row in selected if row["existing_source_path"] is None
            ),
            "files": selected,
        },
        "task": {
            "name": "synchronized wrist-video to wrist-IMU four-choice correspondence",
            "input": {
                "video_query": (
                    "8 RGB frames uniformly sampled from a 2.0-second wrist-left video window"
                ),
                "options": (
                    "four six-channel wrist-left IMU windows, "
                    "each 2.0 seconds and resampled to 128 points"
                ),
                "text": "none; option identifiers are opaque and permuted",
                "metadata_hidden_from_model": [
                    "source path",
                    "group_id",
                    "sequence_id",
                    "category",
                    "timestamps",
                    "window index",
                ],
            },
            "target": (
                "the candidate IMU window whose source-clock interval "
                "matches the video query interval"
            ),
            "negative_options": (
                "three same-duration windows from the recording, each at least 0.5 seconds "
                "from the query interval; "
                "sampled deterministically from other valid windows"
            ),
            "query_windows": (
                "2.0 seconds, 1.0-second stride; omit queries without three valid negatives"
            ),
            "random_seed": 17,
            "primary_metrics": ["four-choice accuracy", "NLL", "Brier", "ECE"],
            "controls": [
                "question/option-only",
                "video-only",
                "IMU-options-only",
                "joint",
                "video/IMU mismatch",
                "missing-video",
                "missing-IMU",
                "option permutation",
            ],
            "model": {
                "video_encoder": (
                    "frozen VideoPrism-B; checkpoint SHA-256 "
                    "61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29"
                ),
                "sensor_encoder": (
                    "frozen MOMENT-1-small; checkpoint SHA-256 "
                    "785e6c6f57ffa7cac7e2a1fff6369618d49f2f441563ddea76c866231e5aa877"
                ),
                "fusion": (
                    "small trainable pair scorer over frozen features; "
                    "normalized probabilities over candidate IDs"
                ),
            },
        },
        "nonclaims": [
            "not activity recognition or a general world-model quality result",
            "not participant-independent validation",
            "not sealed audit",
            "not evidence that all product modalities meet their quality target",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.index, args.split)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
