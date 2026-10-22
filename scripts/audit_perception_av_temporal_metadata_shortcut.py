"""Audit whether train split timestamps alone reveal the AV temporal-match target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _interval(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must contain exactly two endpoints")
    start, end = (float(part) for part in value)
    if not (math.isfinite(start) and math.isfinite(end)) or end < start:
        raise ValueError(f"{field} must be a finite increasing interval")
    return start, end


def audit_train_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    """Measure simple interval-only rules on train rows; never inspect validation."""
    rows = manifest["train"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("train must be a nonempty list")

    seen_video_ids: set[str] = set()
    exact_target = 0
    nearest_center_correct = 0
    nearest_center_ties = 0
    maximum_overlap_correct = 0
    maximum_overlap_ties = 0

    for row in rows:
        video_id = str(row["video_id"])
        if video_id in seen_video_ids:
            raise ValueError("train must contain one row per video group")
        seen_video_ids.add(video_id)

        video_interval = _interval(row["video_window_s"], "video_window_s")
        options = row["options"]
        if not isinstance(options, list) or len(options) != 2:
            raise ValueError("each train row must have exactly two audio options")
        target = int(row["target_index"])
        if target not in (0, 1):
            raise ValueError("target_index must be 0 or 1")

        audio_intervals = [
            _interval(option["audio_window_s"], "audio_window_s") for option in options
        ]
        if audio_intervals[target] == video_interval:
            exact_target += 1

        video_center = sum(video_interval) / 2
        center_gaps = [abs(sum(interval) / 2 - video_center) for interval in audio_intervals]
        if math.isclose(center_gaps[0], center_gaps[1], rel_tol=0.0, abs_tol=1e-12):
            nearest_center_ties += 1
        elif center_gaps.index(min(center_gaps)) == target:
            nearest_center_correct += 1

        overlaps = [
            max(
                0.0,
                min(video_interval[1], interval[1])
                - max(video_interval[0], interval[0]),
            )
            for interval in audio_intervals
        ]
        if math.isclose(overlaps[0], overlaps[1], rel_tol=0.0, abs_tol=1e-12):
            maximum_overlap_ties += 1
        elif overlaps.index(max(overlaps)) == target:
            maximum_overlap_correct += 1

    count = len(rows)
    return {
        "status": "PASS_TRAIN_ONLY_TIMESTAMP_SHORTCUT_AUDIT",
        "split": "train_only",
        "example_count": count,
        "unique_video_groups": len(seen_video_ids),
        "exact_interval_matches": exact_target,
        "exact_interval_match_rate": exact_target / count,
        "nearest_center_correct": nearest_center_correct,
        "nearest_center_ties": nearest_center_ties,
        "nearest_center_accuracy_lower_bound": nearest_center_correct / count,
        "maximum_overlap_correct": maximum_overlap_correct,
        "maximum_overlap_ties": maximum_overlap_ties,
        "maximum_overlap_accuracy_lower_bound": maximum_overlap_correct / count,
        "validation_rows_used": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = audit_train_metadata(manifest)
    result["manifest_sha256"] = _sha256(manifest_path)
    result["manifest_path"] = str(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
