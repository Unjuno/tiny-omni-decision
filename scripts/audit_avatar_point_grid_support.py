"""Measure train-only AVATAR point-grid label support without opening media."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

try:
    from .audit_avatar_multientity_exposure import (
        _sha256_file,
        _source_group_from_annotation,
        collect_prior_group_usage,
    )
    from .audit_avatar_task_support import TASKS, _group_split
except ImportError:  # Direct execution as ``python scripts/audit_avatar_point_grid_support.py``.
    from audit_avatar_multientity_exposure import (
        _sha256_file,
        _source_group_from_annotation,
        collect_prior_group_usage,
    )
    from audit_avatar_task_support import TASKS, _group_split


def _polygons_for_visible_annotations(annotations: list[dict[str, Any]]) -> list[list[float]]:
    polygons: list[list[float]] = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if segmentation is None:
            continue
        if not isinstance(segmentation, list):
            raise ValueError("visible annotation segmentation must be a polygon list")
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 2 or len(polygon) % 2:
                raise ValueError("segmentation polygon must contain at least one xy point")
            if any(
                not isinstance(value, (int, float)) or not math.isfinite(value) for value in polygon
            ):
                raise ValueError("segmentation polygon coordinates must be finite numbers")
            polygons.append(polygon)
    return polygons


def _rasterize_union(width: int, height: int, polygons: list[list[float]]) -> Image.Image:
    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        vertices = list(zip(polygon[0::2], polygon[1::2], strict=True))
        integer_vertices = [(int(x), int(y)) for x, y in vertices]
        if len(integer_vertices) == 1:
            draw.point(integer_vertices[0], fill=1)
        elif len(integer_vertices) == 2:
            draw.line(integer_vertices, fill=1, width=1)
        else:
            draw.polygon(integer_vertices, fill=1)
    return mask


def _grid_positive_count(mask: Image.Image, side: int) -> int:
    width, height = mask.size
    return sum(
        bool(mask.getpixel((int((column + 0.5) * width / side), int((row + 0.5) * height / side))))
        for row in range(side)
        for column in range(side)
    )


def audit_point_grid_support(
    train_annotations: Path,
    *,
    artifact_root: Path,
    seed: int = 17,
    validation_fraction: float = 0.4,
    grid_sizes: tuple[int, ...] = (14, 16, 28),
) -> dict[str, Any]:
    """Evaluate fixed-grid center coverage on globally assigned TRAIN groups only."""
    root = Path(train_annotations).resolve()
    if root.name.lower() != "train" or any(part.lower() == "audit" for part in root.parts):
        raise ValueError("train_annotations must be the isolated TRAIN directory")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not grid_sizes or any(not isinstance(side, int) or side < 2 for side in grid_sizes):
        raise ValueError("grid_sizes must contain positive integer side lengths >= 2")

    fit_groups, evaluation_groups, prior_hashes, prior_rows = collect_prior_group_usage(
        Path(artifact_root)
    )
    previously_exposed = fit_groups | evaluation_groups

    rows: list[tuple[Path, dict[str, Any], str]] = []
    all_unexposed_groups: set[str] = set()
    train_hash = hashlib.sha256()
    for path in sorted(root.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        group = _source_group_from_annotation(path, row)
        relative = path.relative_to(root).as_posix()
        train_hash.update(relative.encode("utf-8") + b"\0")
        train_hash.update(_sha256_file(path).encode("ascii") + b"\n")
        annotations = row.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(f"missing annotations in {path}")
        if any(not isinstance(annotation, dict) for annotation in annotations):
            raise ValueError(f"invalid annotation in {path}")
        tasks = {annotation.get("task") for annotation in annotations}
        if tasks - TASKS:
            raise ValueError(f"unknown AVATAR task in {path}")
        if group not in previously_exposed:
            all_unexposed_groups.add(group)
        if "Mixed-Sound" in tasks:
            rows.append((path, row, group))

    global_fit_groups, global_validation_groups = _group_split(
        all_unexposed_groups, seed, validation_fraction
    )
    counts = {side: Counter() for side in grid_sizes}
    positive_cells_by_grid = {side: [] for side in grid_sizes}
    categories: set[str] = set()
    clips: set[str] = set()
    group_frames: Counter[str] = Counter()
    image_sizes: Counter[tuple[int, int]] = Counter()
    train_mixed_groups: set[str] = set()
    invalid_or_missing_visible_masks = 0

    for path, row, group in rows:
        if group in previously_exposed or group not in global_fit_groups:
            continue
        width, height = row.get("original_width"), row.get("original_height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"invalid original frame dimensions in {path}")
        annotations = row["annotations"]
        mixed = [a for a in annotations if a.get("task") == "Mixed-Sound"]
        if not mixed:
            continue
        polygons = _polygons_for_visible_annotations(annotations)
        if not polygons:
            invalid_or_missing_visible_masks += 1
            continue
        mask = _rasterize_union(width, height, polygons)
        train_mixed_groups.add(group)
        clips.add(row["video_id"])
        group_frames[group] += 1
        image_sizes[(width, height)] += 1
        categories.update(a["audio_visual_category"] for a in mixed)

        for side in grid_sizes:
            positives = _grid_positive_count(mask, side)
            counts[side]["total_queries"] += side * side
            counts[side]["positive_queries"] += positives
            counts[side]["frames_with_positive_query"] += positives > 0
            counts[side]["frames_with_multiple_positive_queries"] += positives > 1
            positive_cells_by_grid[side].append(positives)

    grid_results: dict[str, Any] = {}
    for side in sorted(set(grid_sizes)):
        values = sorted(positive_cells_by_grid[side])
        n = len(values)
        positive = counts[side]["positive_queries"]
        grid_results[str(side)] = {
            "cells_per_frame": side * side,
            "train_frame_count": n,
            "total_binary_queries": counts[side]["total_queries"],
            "positive_queries": positive,
            "negative_queries": counts[side]["total_queries"] - positive,
            "positive_query_fraction": positive / counts[side]["total_queries"]
            if counts[side]["total_queries"]
            else 0.0,
            "frames_with_at_least_one_positive": counts[side]["frames_with_positive_query"],
            "frames_with_multiple_positive_queries": counts[side][
                "frames_with_multiple_positive_queries"
            ],
            "frame_positive_coverage": counts[side]["frames_with_positive_query"] / n if n else 0.0,
            "mean_positive_cells_per_frame": sum(values) / n if n else 0.0,
            "median_positive_cells_per_frame": values[n // 2] if n else 0,
            "p95_positive_cells_per_frame": values[min(n - 1, math.ceil(n * 0.95) - 1)] if n else 0,
        }

    return {
        "format": "avatar_point_grid_train_support_v1",
        "dataset_revision": "a7f0146ff704799d044133e61137836cacc1c233",
        "train_annotation_tree_sha256": train_hash.hexdigest(),
        "prior_exposed_group_set_sha256": hashlib.sha256(
            "\n".join(sorted(previously_exposed)).encode("utf-8")
        ).hexdigest(),
        "prior_sidecar_sha256": prior_hashes,
        "prior_sidecar_row_counts": prior_rows,
        "global_group_split": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "assignment_unit": "source_group_global_across_tasks",
            "unexposed_groups": len(all_unexposed_groups),
            "train_groups": len(global_fit_groups),
            "validation_groups": len(global_validation_groups),
            "train_group_set_sha256": hashlib.sha256(
                "\n".join(sorted(global_fit_groups)).encode("utf-8")
            ).hexdigest(),
            "validation_group_set_sha256": hashlib.sha256(
                "\n".join(sorted(global_validation_groups)).encode("utf-8")
            ).hexdigest(),
        },
        "mixed_sound_train_support": {
            "source_group_count": len(train_mixed_groups),
            "annotated_clip_count": len(clips),
            "annotated_frame_count": sum(group_frames.values()),
            "category_count": len(categories),
            "image_size_distribution": [
                {"width": width, "height": height, "frame_count": count}
                for (width, height), count in sorted(image_sizes.items())
            ],
            "frames_without_visible_union_mask": invalid_or_missing_visible_masks,
        },
        "grid_center_support": grid_results,
        "geometry_method": (
            "Pillow 1-bit polygon rasterization with integer-truncated coordinates, "
            "including one-point and two-point masks. This is a TRAIN-only coverage "
            "approximation, not verified parity with the official OpenCV fillPoly rasterizer."
        ),
        "source_group_ids_emitted": False,
        "official_validation_annotations_accessed": False,
        "internal_validation_group_json_parsed_for_split_replay": True,
        "internal_validation_group_masks_or_dimensions_used": False,
        "sealed_audit_accessed": False,
        "media_payloads_opened_or_decoded": False,
        "model_or_features_loaded": False,
        "quality_result": False,
        "selection_note": (
            "Grid statistics use only globally assigned Mixed-Sound TRAIN groups. "
            "They measure center-sample label coverage, not model quality or "
            "official-resolution localization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.4)
    parser.add_argument("--grid-sizes", default="14,16,28")
    args = parser.parse_args()
    grid_sizes = tuple(int(value) for value in args.grid_sizes.split(","))
    report = audit_point_grid_support(
        args.train_annotations,
        artifact_root=args.artifact_root,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        grid_sizes=grid_sizes,
    )
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report["grid_center_support"], sort_keys=True))


if __name__ == "__main__":
    main()
