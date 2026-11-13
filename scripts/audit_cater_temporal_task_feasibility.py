"""Metadata-only feasibility audit for a CATER temporal Decision task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_COLLISION_SHA256 = "94bf68dc1a56fea3f227a3552a22f6d8e49ecd76b916dd3c4069f91a2b2b6f50"
EXPECTED_SCENES_SHA256 = "2855bc5d765516b113dc3bf1cd8c43c7923d3a6350450af816ac6790f4d4f51d"
EXPECTED_LIST_SHA256 = {
    "train_subsetT.txt": "62b1733632a7f447bcc1fcb1a4e1cb4472fabf14d756fe5cabd5df4afddd23cb",
    "train_subsetV.txt": "e320cfe0af43d8c1256885096952a4a0bb4e31475332f720a6b530cccd425b81",
}
WINDOWS = ((0, 75), (75, 150), (150, 225), (225, 300))
MIN_DISPLACEMENT = 0.20
MIN_WINNER_MARGIN = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distinct_visual_descriptors(objects: list[dict[str, Any]]) -> bool:
    descriptors = [
        (obj.get("shape"), obj.get("color"), obj.get("material"))
        for obj in objects
    ]
    return bool(descriptors) and len(set(descriptors)) == len(descriptors)


def temporal_query_windows(
    scene: dict[str, Any],
    windows: tuple[tuple[int, int], ...] = WINDOWS,
    min_displacement: float = MIN_DISPLACEMENT,
    min_winner_margin: float = MIN_WINNER_MARGIN,
) -> list[tuple[int, int]]:
    """Return fixed intervals with an unambiguous largest object displacement."""
    objects = scene.get("objects", [])
    if not distinct_visual_descriptors(objects):
        return []

    valid = []
    for start, end in windows:
        distances = []
        for obj in objects:
            locations = obj.get("locations", {})
            try:
                first = [float(value) for value in locations[str(start)]]
                last = [float(value) for value in locations[str(end)]]
            except (KeyError, TypeError, ValueError):
                continue
            if len(first) != len(last) or not first:
                continue
            distances.append(math.dist(first, last))
        if len(distances) < 2:
            continue
        distances.sort(reverse=True)
        if (
            distances[0] >= min_displacement
            and distances[0] - distances[1] >= min_winner_margin
        ):
            valid.append((start, end))
    return valid


def parse_video_id(line: str) -> int:
    filename = line.split()[0]
    if not filename.startswith("CATER_new_") or not filename.endswith(".avi"):
        raise ValueError(f"Unexpected CATER list entry: {line!r}")
    return int(filename.removeprefix("CATER_new_").removesuffix(".avi"))


def audit(scenes_zip: Path, lists_dir: Path, collision_receipt: Path) -> dict[str, Any]:
    if sha256_file(scenes_zip) != EXPECTED_SCENES_SHA256:
        raise ValueError("CATER scene archive hash does not match the audited revision")
    if sha256_file(collision_receipt) != EXPECTED_COLLISION_SHA256:
        raise ValueError("Clevr-4 collision receipt hash does not match the audited receipt")

    collision_data = json.loads(collision_receipt.read_text(encoding="utf-8"))
    partitions = {
        "train": ("train_subsetT.txt", "cater_train_subset_t"),
        "validation": ("train_subsetV.txt", "cater_selector_subset_v"),
    }
    split_ids: dict[str, set[int]] = {}
    rows: dict[str, dict[str, Any]] = {}

    with zipfile.ZipFile(scenes_zip) as archive:
        for split, (list_name, collision_key) in partitions.items():
            list_path = lists_dir / list_name
            if sha256_file(list_path) != EXPECTED_LIST_SHA256[list_name]:
                raise ValueError(f"CATER list hash mismatch: {list_name}")
            excluded = set(collision_data[collision_key]["excluded_video_ids"])
            listed_ids = [
                parse_video_id(line)
                for line in list_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(listed_ids) != len(set(listed_ids)):
                raise ValueError(f"Duplicate video ID in {list_name}")
            kept_ids = [video_id for video_id in listed_ids if video_id not in excluded]
            split_ids[split] = set(kept_ids)
            per_window: Counter[str] = Counter()
            option_counts: Counter[str] = Counter()
            eligible_video_ids = set()
            query_count = 0
            missing_scenes = []

            for video_id in kept_ids:
                member = f"scenes/CATER_new_{video_id:06d}.json"
                try:
                    scene = json.loads(archive.read(member))
                except KeyError:
                    missing_scenes.append(video_id)
                    continue
                if scene.get("image_index") != video_id:
                    raise ValueError(f"Scene/video identity mismatch for {video_id}")
                valid_windows = temporal_query_windows(scene)
                for interval in valid_windows:
                    per_window[f"{interval[0]}-{interval[1]}"] += 1
                if valid_windows:
                    eligible_video_ids.add(video_id)
                    query_count += len(valid_windows)
                    option_counts[str(len(scene.get("objects", [])))] += len(
                        valid_windows
                    )

            if missing_scenes:
                raise ValueError(f"Missing scene JSONs in {split}: {missing_scenes[:5]}")
            rows[split] = {
                "listed_video_ids": len(listed_ids),
                "excluded_numeric_id_collisions": len(listed_ids) - len(kept_ids),
                "eligible_video_groups": len(eligible_video_ids),
                "candidate_video_groups": len(kept_ids),
                "unique_temporal_queries": query_count,
                "queries_by_option_count": dict(sorted(option_counts.items())),
                "queries_by_fixed_interval": dict(sorted(per_window.items())),
            }

    overlap = sorted(split_ids["train"] & split_ids["validation"])
    if overlap:
        raise ValueError(f"Train/validation video overlap: {overlap[:5]}")
    return {
        "schema_version": 1,
        "scope": "metadata-only; no video media, model, predictions, or sealed audit",
        "source": {
            "cater_commit": "13a19643f1a2fb24e931df25abd74353e4f2fdcb",
            "scenes_zip_sha256": EXPECTED_SCENES_SHA256,
            "train_list_sha256": EXPECTED_LIST_SHA256["train_subsetT.txt"],
            "selector_list_sha256": EXPECTED_LIST_SHA256["train_subsetV.txt"],
            "clevr4_collision_receipt_sha256": EXPECTED_COLLISION_SHA256,
        },
        "task": {
            "definition": (
                "Select which visually distinguishable object moved the farthest "
                "over one fixed 75-frame interval."
            ),
            "intervals": [list(window) for window in WINDOWS],
            "min_max_displacement_world_units": MIN_DISPLACEMENT,
            "min_winner_margin_world_units": MIN_WINNER_MARGIN,
            "object_identity_fields": ["shape", "color", "material"],
        },
        "train_validation_video_id_overlap": 0,
        "splits": rows,
        "limitations": [
            (
                "Scene trajectories are label/proxy metadata; the rendered video task "
                "still requires media-level visual and temporal validation."
            ),
            (
                "Multiple intervals from one video are correlated; uncertainty must "
                "be grouped by video/scene."
            ),
            (
                "Clevr-4 cross-ID checks exclude numeric collisions and coarse "
                "attribute matches but do not prove absence of visual near-duplicates."
            ),
            "This is feasibility evidence only, not model quality evidence or source admission.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-zip", required=True, type=Path)
    parser.add_argument("--lists-dir", required=True, type=Path)
    parser.add_argument("--collision-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.scenes_zip, args.lists_dir, args.collision_receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in result.items() if key != "limitations"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
