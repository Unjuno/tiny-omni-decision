"""Compare AVATAR's pinned OpenCV union-mask rasterizer with Pillow on TRAIN only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from .audit_avatar_multientity_exposure import (
        _sha256_file,
        _source_group_from_annotation,
        collect_prior_group_usage,
    )
    from .audit_avatar_point_grid_support import (
        _polygons_for_visible_annotations,
        _rasterize_union,
    )
    from .audit_avatar_task_support import TASKS, _group_split
except ImportError:  # Direct execution as ``python scripts/audit_avatar_rasterizer_parity.py``.
    from audit_avatar_multientity_exposure import (
        _sha256_file,
        _source_group_from_annotation,
        collect_prior_group_usage,
    )
    from audit_avatar_point_grid_support import (
        _polygons_for_visible_annotations,
        _rasterize_union,
    )
    from audit_avatar_task_support import TASKS, _group_split


DATASET_REVISION = "a7f0146ff704799d044133e61137836cacc1c233"
OFFICIAL_UTILS_SHA256 = "cef3a03a5b3befff7d1859526ed075a32e7d3ebfba70b51ae3e9f3b94b8d549b"
OFFICIAL_EVALUATOR_SHA256 = "0560545251519b5d0bce20ca1d3c8e2d57b5545f7fe08c85302d99f4f300827b"


def _official_opencv_union_mask(
    width: int,
    height: int,
    annotations: list[dict[str, Any]],
    *,
    cv2: Any,
    np: Any,
) -> Any:
    """Mirror pinned utils.py convert_ann_to_mask + evaluator's logical union."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if segmentation is None:
            continue
        if not isinstance(segmentation, list):
            raise ValueError("visible annotation segmentation must be a polygon list")
        for polygon in segmentation:
            # Match the pinned official utility: np.array(p).reshape(-1, 2).astype(int)
            points = np.array(polygon).reshape(-1, 2).astype(int)
            cv2.fillPoly(mask, [points], 1)
    return mask


def _center_samples(width: int, height: int, side: int) -> tuple[list[int], list[int]]:
    if side < 2:
        raise ValueError("grid side must be >= 2")
    xs = [int((column + 0.5) * width / side) for column in range(side)]
    ys = [int((row + 0.5) * height / side) for row in range(side)]
    return xs, ys


def _compare_masks(
    pillow_mask: Image.Image,
    opencv_mask: Any,
    grid_sizes: tuple[int, ...],
    np: Any,
):
    pillow = np.asarray(pillow_mask, dtype=np.uint8) != 0
    opencv = np.asarray(opencv_mask) != 0
    if pillow.shape != opencv.shape or pillow.ndim != 2:
        raise ValueError("mask dimensions do not match")
    mismatch = pillow != opencv
    height, width = pillow.shape
    grids = {}
    for side in grid_sizes:
        xs, ys = _center_samples(width, height, side)
        pillow_points = pillow[np.ix_(ys, xs)]
        opencv_points = opencv[np.ix_(ys, xs)]
        grids[str(side)] = {
            "queries_per_frame": side * side,
            "pillow_positive_points": int(np.count_nonzero(pillow_points)),
            "opencv_positive_points": int(np.count_nonzero(opencv_points)),
            "different_point_labels": int(np.count_nonzero(pillow_points != opencv_points)),
        }
    return int(np.count_nonzero(mismatch)), int(mismatch.size), grids


def audit_rasterizer_parity(
    train_annotations: Path,
    *,
    artifact_root: Path,
    expected_opencv_version: str,
    seed: int = 17,
    validation_fraction: float = 0.4,
    grid_sizes: tuple[int, ...] = (14, 16, 28),
) -> dict[str, Any]:
    """Compare mask geometry on globally assigned, previously unseen TRAIN groups."""
    import cv2
    import numpy as np

    if cv2.__version__ != expected_opencv_version:
        raise ValueError(
            f"OpenCV version mismatch: expected {expected_opencv_version}, got {cv2.__version__}"
        )
    root = Path(train_annotations).resolve()
    if root.name.lower() != "train" or any(part.lower() == "audit" for part in root.parts):
        raise ValueError("train_annotations must be the isolated TRAIN directory")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if not grid_sizes or any(not isinstance(side, int) or side < 2 for side in grid_sizes):
        raise ValueError("grid_sizes must contain integer side lengths >= 2")

    fit_groups, evaluation_groups, prior_hashes, prior_rows = collect_prior_group_usage(
        Path(artifact_root)
    )
    previously_exposed = fit_groups | evaluation_groups
    rows: list[tuple[Path, dict[str, Any], str]] = []
    unexposed_groups: set[str] = set()
    tree_hash = hashlib.sha256()
    for path in sorted(root.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        group = _source_group_from_annotation(path, row)
        tree_hash.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        tree_hash.update(_sha256_file(path).encode("ascii") + b"\n")
        annotations = row.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(f"missing annotations in {path}")
        if any(not isinstance(annotation, dict) for annotation in annotations):
            raise ValueError(f"invalid annotation in {path}")
        tasks = {annotation.get("task") for annotation in annotations}
        if tasks - TASKS:
            raise ValueError(f"unknown AVATAR task in {path}")
        if group not in previously_exposed:
            unexposed_groups.add(group)
        if "Mixed-Sound" in tasks:
            rows.append((path, row, group))

    train_groups, validation_groups = _group_split(unexposed_groups, seed, validation_fraction)
    pixel_mismatches = 0
    total_pixels = 0
    frames_with_pixel_difference = 0
    frame_count = 0
    grid_counts = {
        side: Counter(
            pillow_positive_points=0,
            opencv_positive_points=0,
            different_point_labels=0,
            frames_with_different_point_labels=0,
            frames_with_pillow_positive=0,
            frames_with_opencv_positive=0,
        )
        for side in grid_sizes
    }
    categories: set[str] = set()
    clips: set[str] = set()
    groups: set[str] = set()
    for path, row, group in rows:
        if group in previously_exposed or group not in train_groups:
            continue
        width, height = row.get("original_width"), row.get("original_height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"invalid image dimensions in {path}")
        annotations = row["annotations"]
        mixed = [a for a in annotations if a.get("task") == "Mixed-Sound"]
        if not mixed:
            continue
        polygons = _polygons_for_visible_annotations(annotations)
        if not polygons:
            raise ValueError(f"Mixed-Sound TRAIN frame has no visible polygon: {path}")
        pillow_mask = _rasterize_union(width, height, polygons)
        opencv_mask = _official_opencv_union_mask(width, height, annotations, cv2=cv2, np=np)
        mismatch, pixels, grid_result = _compare_masks(pillow_mask, opencv_mask, grid_sizes, np)
        pixel_mismatches += mismatch
        total_pixels += pixels
        frames_with_pixel_difference += mismatch > 0
        frame_count += 1
        groups.add(group)
        clips.add(row["video_id"])
        categories.update(a["audio_visual_category"] for a in mixed)
        for side in grid_sizes:
            item = grid_result[str(side)]
            counts = grid_counts[side]
            for key in (
                "pillow_positive_points",
                "opencv_positive_points",
                "different_point_labels",
            ):
                counts[key] += item[key]
            counts["frames_with_different_point_labels"] += item["different_point_labels"] > 0
            counts["frames_with_pillow_positive"] += item["pillow_positive_points"] > 0
            counts["frames_with_opencv_positive"] += item["opencv_positive_points"] > 0

    grid_report = {}
    for side in sorted(set(grid_sizes)):
        counts = grid_counts[side]
        queries = frame_count * side * side
        grid_report[str(side)] = {
            "queries_per_frame": side * side,
            "total_queries": queries,
            "pillow_positive_points": counts["pillow_positive_points"],
            "opencv_positive_points": counts["opencv_positive_points"],
            "different_point_labels": counts["different_point_labels"],
            "different_point_label_fraction": (
                counts["different_point_labels"] / queries if queries else 0.0
            ),
            "frames_with_different_point_labels": counts["frames_with_different_point_labels"],
            "frames_with_pillow_positive": counts["frames_with_pillow_positive"],
            "frames_with_opencv_positive": counts["frames_with_opencv_positive"],
        }

    return {
        "format": "avatar_pillow_opencv_train_mask_parity_v1",
        "dataset_revision": DATASET_REVISION,
        "official_code": {
            "utils_path": "code/evaluation/utils.py",
            "utils_sha256": OFFICIAL_UTILS_SHA256,
            "evaluator_path": "code/evaluation/evaluator.py",
            "evaluator_sha256": OFFICIAL_EVALUATOR_SHA256,
            "semantics": (
                "cv2.fillPoly per polygon, then logical union across visible instance masks"
            ),
        },
        "rasterizer_comparison": {
            "pillow": "ImageDraw 1-bit polygons, integer-truncated coordinates",
            "opencv_version": cv2.__version__,
            "official_conversion": (
                "np.array(p).reshape(-1, 2).astype(int) then "
                "cv2.fillPoly(mask, [p], 1)"
            ),
            "frame_count": frame_count,
            "raster_pixels_compared": total_pixels,
            "different_pixels": pixel_mismatches,
            "different_pixel_fraction": pixel_mismatches / total_pixels if total_pixels else 0.0,
            "frames_with_any_pixel_difference": frames_with_pixel_difference,
        },
        "global_group_split": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "assignment_unit": "source_group_global_across_tasks",
            "previously_unexposed_groups": len(unexposed_groups),
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
            "train_group_set_sha256": hashlib.sha256(
                "\n".join(sorted(train_groups)).encode("utf-8")
            ).hexdigest(),
            "validation_group_set_sha256": hashlib.sha256(
                "\n".join(sorted(validation_groups)).encode("utf-8")
            ).hexdigest(),
        },
        "train_support": {
            "source_groups": len(groups),
            "clips": len(clips),
            "frames": frame_count,
            "categories": len(categories),
        },
        "point_grid_comparison": grid_report,
        "prior_sidecar_sha256": prior_hashes,
        "prior_sidecar_row_counts": prior_rows,
        "train_annotation_tree_sha256": tree_hash.hexdigest(),
        "source_group_ids_emitted": False,
        "official_validation_annotations_accessed": False,
        "internal_validation_group_json_parsed_for_split_replay": True,
        "internal_validation_group_masks_or_dimensions_used": False,
        "sealed_audit_accessed": False,
        "media_payloads_opened_or_decoded": False,
        "model_or_features_loaded": False,
        "quality_result": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotations", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-opencv-version", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.4)
    parser.add_argument("--grid-sizes", default="14,16,28")
    args = parser.parse_args()
    report = audit_rasterizer_parity(
        args.train_annotations,
        artifact_root=args.artifact_root,
        expected_opencv_version=args.expected_opencv_version,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        grid_sizes=tuple(int(value) for value in args.grid_sizes.split(",")),
    )
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report["rasterizer_comparison"], sort_keys=True))
    print(json.dumps(report["point_grid_comparison"], sort_keys=True))


if __name__ == "__main__":
    main()
