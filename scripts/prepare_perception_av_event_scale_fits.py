"""Build old-only and expanded-fit manifests for the fixed-budget comparison."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_perception_av_correspondence import sha256_file
from scripts.av_event_split_validation import validate_video_splits

OLD_FEATURE_MANIFEST_SHA256 = "d92e4c91c236c69548f8fa0b58826e8afc853372ebf7a59b1e64567034f8df57"
SCALE_PLAN_SHA256 = "43dda454adf45352dee281680f51698ecf38d60731d33311a6ded139de167461"
HOLDOUT_PLAN_SHA256 = "887a1eea4b5a7b82d441025842465a09d0ffb33c97787073042ad1a1680f8eb3"
OLD_FIT_EVENTS = 525
OLD_FIT_VIDEOS = 240
NEW_SPLIT_EVENT_COUNTS = {
    "fit": 550,
    "selection": 121,
    "calibration": 91,
    "development_validation": 146,
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"feature manifest is empty or malformed: {path}")
    return rows


def build_comparison_rows(
    old_rows: list[dict], new_rows: list[dict], holdout_video_ids: set[str]
) -> dict[str, list[dict]]:
    old_fit = [row for row in old_rows if row.get("split") == "fit"]
    if (
        len(old_fit) != OLD_FIT_EVENTS
        or len({row["video_id"] for row in old_fit}) != OLD_FIT_VIDEOS
    ):
        raise ValueError("old-only reference must be the frozen 240-video / 525-event fit split")
    if Counter(row.get("split") for row in new_rows) != Counter(
        {split: count for split, count in NEW_SPLIT_EVENT_COUNTS.items()}
    ):
        raise ValueError("new scale feature split counts differ from the frozen plan")

    old_video_ids = {row["video_id"] for row in old_fit}
    new_video_ids = {row["video_id"] for row in new_rows}
    old_media_ids = {row["media_sha256"] for row in old_fit}
    new_media_ids = {row["media_sha256"] for row in new_rows}
    for name, rows in (("old fit", old_fit), ("new scale", new_rows)):
        video_media: dict[str, str] = {}
        media_video: dict[str, str] = {}
        for row in rows:
            video_id = row.get("video_id")
            media_hash = row.get("media_sha256")
            if (
                not isinstance(video_id, str)
                or not isinstance(media_hash, str)
                or len(media_hash) != 64
                or any(char not in "0123456789abcdef" for char in media_hash)
            ):
                raise ValueError(f"{name} row lacks a valid video/media identity")
            if video_media.setdefault(video_id, media_hash) != media_hash:
                raise ValueError(f"{name} video maps to multiple media identities")
            if media_video.setdefault(media_hash, video_id) != video_id:
                raise ValueError(f"{name} media identity maps to multiple video IDs")
    if old_video_ids & new_video_ids or old_media_ids & new_media_ids:
        raise ValueError("old and new fit corpora have overlapping video/media identities")
    if holdout_video_ids & (old_video_ids | new_video_ids):
        raise ValueError("fixed-checkpoint holdout video IDs entered the fit manifests")

    new_holdouts = [row for row in new_rows if row["split"] != "fit"]
    manifests = {
        "old_only": sorted(old_fit + new_holdouts, key=lambda row: row["id"]),
        "expanded": sorted(old_fit + new_rows, key=lambda row: row["id"]),
    }
    for rows in manifests.values():
        validate_video_splits(rows)
        if not (
            any(row["split"] == "fit" for row in rows)
            and any(row["split"] == "selection" for row in rows)
            and any(row["split"] == "calibration" for row in rows)
            and any(row["split"] == "development_validation" for row in rows)
        ):
            raise ValueError("comparison manifest lacks a required video-disjoint split")
    return manifests


def prepare(
    old_manifest: Path,
    new_manifest: Path,
    new_result: Path,
    scale_plan: Path,
    holdout_plan: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise ValueError("fresh comparison-manifest output required")
    old_hash = sha256_file(old_manifest)
    if old_hash != OLD_FEATURE_MANIFEST_SHA256:
        raise ValueError("old reference feature manifest hash changed")
    if sha256_file(scale_plan) != SCALE_PLAN_SHA256:
        raise ValueError("scale plan hash changed")
    scale_plan_data = json.loads(scale_plan.read_text(encoding="utf-8"))
    if sha256_file(holdout_plan) != HOLDOUT_PLAN_SHA256:
        raise ValueError("fixed-checkpoint holdout plan hash changed")
    holdout = json.loads(holdout_plan.read_text(encoding="utf-8"))
    if len(holdout.get("rows", [])) != 200:
        raise ValueError("fixed-checkpoint holdout plan is malformed")
    feature_result = json.loads(new_result.read_text(encoding="utf-8"))
    new_hash = sha256_file(new_manifest)
    if (
        feature_result.get("status") != "PASS_FROZEN_VIDEO_SCALE_FEATURES"
        or feature_result.get("plan_sha256") != SCALE_PLAN_SHA256
        or feature_result.get("feature_manifest_sha256") != new_hash
        or feature_result.get("event_count") != scale_plan_data["candidate_events"]
        or feature_result.get("split_event_counts") != NEW_SPLIT_EVENT_COUNTS
    ):
        raise ValueError("new feature receipt differs from scale plan or manifest")
    old_rows = _read_jsonl(old_manifest)
    new_rows = _read_jsonl(new_manifest)
    manifests = build_comparison_rows(
        old_rows,
        new_rows,
        {row["video_id"] for row in holdout["rows"]},
    )

    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for name, rows in manifests.items():
        path = output / f"{name}-features.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        split_events = dict(Counter(row["split"] for row in rows))
        split_videos = {
            split: len({row["video_id"] for row in rows if row["split"] == split})
            for split in sorted(split_events)
        }
        summaries[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "event_count": len(rows),
            "video_count": len({row["video_id"] for row in rows}),
            "split_event_counts": split_events,
            "split_video_counts": split_videos,
        }
    result = {
        "status": "PASS_AV_EVENT_SCALE_COMPARISON_MANIFESTS",
        "old_feature_manifest_sha256": old_hash,
        "new_feature_manifest_sha256": new_hash,
        "scale_plan_sha256": SCALE_PLAN_SHA256,
        "comparison_manifests": summaries,
        "old_fit_events": len([row for row in old_rows if row["split"] == "fit"]),
        "new_unique_fit_events": NEW_SPLIT_EVENT_COUNTS["fit"],
        "fixed_checkpoint_holdout_used": False,
        "training": False,
        "gpu_used": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-features", type=Path, required=True)
    parser.add_argument("--new-features", type=Path, required=True)
    parser.add_argument("--new-result", type=Path, required=True)
    parser.add_argument("--scale-plan", type=Path, required=True)
    parser.add_argument("--holdout-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.old_features,
                args.new_features,
                args.new_result,
                args.scale_plan,
                args.holdout_plan,
                args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
