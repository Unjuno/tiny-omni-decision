"""Build a deterministic, group-safe AVATAR audio-to-video option-choice set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

CLIP_ID = re.compile(r"([A-Za-z0-9_-]{11})_(\d+)\Z")
AUDIO_WINDOW_SECONDS = 2.0


def _rank(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_single_sound_rows(root: Path, split: str) -> list[dict]:
    rows = []
    for path in sorted((root / split).rglob("*.json")):
        clip_id = path.parent.name
        match = CLIP_ID.fullmatch(clip_id)
        if match is None:
            raise ValueError(f"invalid AVATAR clip folder: {clip_id}")
        source_id, offset = match.groups()
        frame = json.loads(path.read_text(encoding="utf-8"))
        if frame.get("video_id") != f"{source_id}_{int(offset)}":
            raise ValueError(f"video_id does not match clip folder: {clip_id}")
        annotations = frame.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"missing annotations in {path}")
        if len(annotations) != 1 or annotations[0].get("task") != "Single-Sound":
            continue
        annotation = annotations[0]
        category = annotation.get("audio_visual_category")
        bbox = annotation.get("bbox")
        if not isinstance(category, str) or not category:
            raise ValueError(f"missing category in {path}")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Single-Sound frame has no usable bbox in {path}")
        frame_number = frame.get("frame_number")
        fps = frame.get("fps")
        duration = frame.get("duration")
        if (
            not isinstance(frame_number, int)
            or not isinstance(fps, (int, float))
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError(f"missing frame timestamp in {path}")
        if (
            not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(f"missing media duration in {path}")
        asset_key = f"{clip_id}/{frame_number:05d}"
        rows.append(
            {
                "asset_key": asset_key,
                "clip_id": clip_id,
                "source_group": source_id,
                "frame_number": frame_number,
                "fps": float(fps),
                "timestamp_seconds": frame_number / float(fps),
                "duration_seconds": float(duration),
                "bbox": bbox,
                "category": category,
            }
        )
    return rows


def _build_split(
    rows: list[dict], *, seed: int, split: str, cross_source_positive: bool = False
) -> tuple[list[dict], list[dict], dict]:
    pools: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pools[row["category"]].append(row)
    for category in pools:
        pools[category].sort(key=lambda row: row["asset_key"])

    model_examples: list[dict] = []
    analysis_rows: list[dict] = []
    target_positions: Counter[int] = Counter()
    padded_audio_windows = 0
    skipped_no_negative_set = 0
    for target in rows:
        available_categories = [
            category
            for category, pool in pools.items()
            if category != target["category"]
            and any(candidate["source_group"] != target["source_group"] for candidate in pool)
        ]
        available_categories.sort(
            key=lambda category: _rank(seed, split, target["asset_key"], category)
        )

        positives = [target]
        if cross_source_positive:
            positives = [
                candidate
                for candidate in pools[target["category"]]
                if candidate["source_group"] != target["source_group"]
                and candidate["asset_key"] != target["asset_key"]
                and abs(candidate["timestamp_seconds"] - target["timestamp_seconds"]) > 1e-9
            ]
            positives.sort(
                key=lambda row: _rank(
                    seed, split, target["asset_key"], "positive", row["asset_key"]
                )
            )
            if not positives:
                skipped_no_negative_set += 1
                continue
        positive = positives[0]

        negatives: list[dict] = []
        used_groups = {target["source_group"]}
        if positive["source_group"] != target["source_group"]:
            used_groups.add(positive["source_group"])
        for category in available_categories:
            pool = [
                candidate
                for candidate in pools[category]
                if candidate["source_group"] not in used_groups
            ]
            if not pool:
                continue
            pool.sort(
                key=lambda row: _rank(
                    seed, split, target["asset_key"], category, row["asset_key"]
                )
            )
            selected = pool[0]
            negatives.append(selected)
            used_groups.add(selected["source_group"])
            if len(negatives) == 3:
                break
        if len(negatives) != 3:
            skipped_no_negative_set += 1
            continue

        candidates = [positive, *negatives]
        candidates.sort(
            key=lambda row: _rank(
                seed, split, target["asset_key"], "option", row["asset_key"]
            )
        )
        option_ids = [f"option_{index}" for index in range(4)]
        target_index = next(
            index for index, candidate in enumerate(candidates) if candidate is positive
        )
        example_id = hashlib.sha256(
            f"avatar-av-choice-v1:{seed}:{split}:{target['asset_key']}".encode()
        ).hexdigest()[:24]
        left_padding = max(0.0, AUDIO_WINDOW_SECONDS / 2 - target["timestamp_seconds"])
        right_padding = max(
            0.0,
            target["timestamp_seconds"]
            + AUDIO_WINDOW_SECONDS / 2
            - target["duration_seconds"],
        )
        padded_audio_windows += int(left_padding > 0 or right_padding > 0)
        model_examples.append(
            {
                "example_id": example_id,
                "split": split,
                "audio_asset_key": target["asset_key"],
                "audio_window_center_seconds": target["timestamp_seconds"],
                "audio_window_duration_seconds": AUDIO_WINDOW_SECONDS,
                "audio_window_left_padding_seconds": left_padding,
                "audio_window_right_padding_seconds": right_padding,
                "options": [
                    {
                        "option_id": option_ids[index],
                        "video_asset_key": candidate["asset_key"],
                        "video_timestamp_seconds": candidate["timestamp_seconds"],
                    }
                    for index, candidate in enumerate(candidates)
                ],
                "target_option_id": option_ids[target_index],
            }
        )
        sidecar = {
            "example_id": example_id,
            "target_source_group": positive["source_group"],
            "target_category": target["category"],
            "target_option_index": target_index,
            "option_source_groups": [candidate["source_group"] for candidate in candidates],
            "option_categories": [candidate["category"] for candidate in candidates],
            "option_asset_keys": [candidate["asset_key"] for candidate in candidates],
            "option_timestamps_seconds": [
                candidate["timestamp_seconds"] for candidate in candidates
            ],
        }
        if cross_source_positive:
            sidecar["audio_source_group"] = target["source_group"]
        analysis_rows.append(sidecar)
        target_positions[target_index] += 1

    report = {
        "single_sound_source_rows": len(rows),
        "generated_choice_rows": len(model_examples),
        "audio_windows_requiring_zero_padding": padded_audio_windows,
        "skipped_without_three_distinct_negative_categories_and_groups": skipped_no_negative_set,
        "unique_target_assets": len({row["audio_asset_key"] for row in model_examples}),
        "unique_target_source_groups": len({row["target_source_group"] for row in analysis_rows}),
        "unique_option_assets": len(
            {option["video_asset_key"] for row in model_examples for option in row["options"]}
        ),
        "target_option_position_counts": {str(i): target_positions[i] for i in range(4)},
        "model_input_contains_category_text": False,
        "candidate_categories_unique_within_row": all(
            len(row["option_categories"]) == len(set(row["option_categories"]))
            for row in analysis_rows
        ),
        "candidate_source_groups_unique_within_row": all(
            len(row["option_source_groups"]) == len(set(row["option_source_groups"]))
            for row in analysis_rows
        ),
    }
    if cross_source_positive:
        report.update(
            {
                "unique_audio_source_groups": len(
                    {row["audio_source_group"] for row in analysis_rows}
                ),
                "positive_candidate_crosses_audio_source_group": True,
                "positive_candidate_timestamp_equals_audio_center_count": sum(
                    abs(
                        example["audio_window_center_seconds"]
                        - example["options"][
                            int(example["target_option_id"].rsplit("_", 1)[1])
                        ]["video_timestamp_seconds"]
                    )
                    <= 1e-9
                    for example in model_examples
                ),
                "positive_candidate_asset_equals_audio_asset_count": sum(
                    example["audio_asset_key"]
                    == example["options"][
                        int(example["target_option_id"].rsplit("_", 1)[1])
                    ]["video_asset_key"]
                    for example in model_examples
                ),
            }
        )
    return model_examples, analysis_rows, report


def build_choices(root: Path, *, seed: int = 17, cross_source_positive: bool = False) -> dict:
    root = Path(root)
    train_rows = _load_single_sound_rows(root, "train")
    validation_rows = _load_single_sound_rows(root, "validation")
    train_groups = {row["source_group"] for row in train_rows}
    validation_groups = {row["source_group"] for row in validation_rows}
    if train_groups & validation_groups:
        raise ValueError("train/validation source-video groups overlap")

    train_model, train_analysis, train_report = _build_split(
        train_rows, seed=seed, split="train", cross_source_positive=cross_source_positive
    )
    validation_model, validation_analysis, validation_report = _build_split(
        validation_rows,
        seed=seed,
        split="validation",
        cross_source_positive=cross_source_positive,
    )
    result = {
        "format": (
            "avatar-audio-video-option-choice-preflight-v2"
            if cross_source_positive
            else "avatar-audio-video-option-choice-preflight-v1"
        ),
        "seed": seed,
        "split_group_key": "source YouTube ID encoded in clip folder",
        "task": (
            "match a Single-Sound audio segment to the same category in a different source video"
            if cross_source_positive
            else "match a Single-Sound audio segment to one of four video event windows"
        ),
        "options_per_example": 4,
        "negative_policy": (
            "three distinct annotated sound categories from three distinct source groups "
            "within the same split"
        ),
        "model_input_fields": [
            "2-second audio crop centered at the annotated frame timestamp",
            "zero-padding when the centered audio window crosses a clip boundary",
            "four video frames at their annotated timestamps",
            "four opaque option IDs",
        ],
        "analysis_sidecar_fields_excluded_from_model": [
            "target_category",
            "target_source_group",
            "option_categories",
            "option_source_groups",
            "option_asset_keys",
        ],
        "splits": {
            "train": {"examples": train_model, "analysis": train_analysis, "report": train_report},
            "validation": {
                "examples": validation_model,
                "analysis": validation_analysis,
                "report": validation_report,
            },
        },
    }
    if cross_source_positive:
        result["cross_source_positive"] = True
        result["positive_policy"] = (
            "same annotated category from a different source group and different timestamp"
        )
        result["analysis_sidecar_fields_excluded_from_model"].append("audio_source_group")
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=17, type=int)
    parser.add_argument("--cross-source-positive", action="store_true")
    args = parser.parse_args()
    result = build_choices(
        args.annotations_root, seed=args.seed, cross_source_positive=args.cross_source_positive
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, values in result["splits"].items():
        write_jsonl(args.output_dir / f"{split}-model-input.jsonl", values["examples"])
        write_jsonl(args.output_dir / f"{split}-analysis-sidecar.jsonl", values["analysis"])
    report = {key: value for key, value in result.items() if key != "splits"}
    report["splits"] = {split: values["report"] for split, values in result["splits"].items()}
    with (args.output_dir / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
