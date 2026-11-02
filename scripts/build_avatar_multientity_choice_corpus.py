"""Freeze a group-disjoint AVATAR TRAIN audio-to-video choice corpus.

The task matches an audio query to one of four video-frame options carrying the
same annotated AV category. Only source TRAIN annotations are read. Prior-use
groups and the sealed audit are excluded; outputs are private run artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.audit_avatar_multientity_exposure import (
    _sha256_file,
    _sha256_ids,
    _source_group_from_annotation,
    collect_prior_group_usage,
    provisional_group_split,
)

WINDOW_SECONDS = 2.0
SEED = 17
CLIP_ID = re.compile(r"([A-Za-z0-9_-]{11})_(\d+)\Z")


def _rank(*parts: str) -> str:
    return hashlib.sha256("\0".join((str(SEED), *parts)).encode()).hexdigest()


def _tree_hash(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode() + b"\0")
        digest.update(_sha256_file(path).encode("ascii") + b"\n")
        count += 1
    return digest.hexdigest(), count


def _load_multientity_assets(train_annotations: Path) -> tuple[dict[str, list[dict]], set[str]]:
    assets_by_group: dict[str, list[dict]] = defaultdict(list)
    groups: set[str] = set()
    for path in sorted(train_annotations.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        group = _source_group_from_annotation(path, row)
        annotations = [
            annotation
            for annotation in row.get("annotations", [])
            if isinstance(annotation, dict)
            and annotation.get("task") == "Multi-Entity"
            and annotation.get("audio_visual_category")
        ]
        if not annotations:
            continue
        video_id = row.get("video_id")
        frame_number = row.get("frame_number")
        fps = row.get("fps")
        duration = row.get("duration")
        if (
            not isinstance(video_id, str)
            or not isinstance(frame_number, int)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or fps <= 0
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration <= 0
        ):
            raise ValueError(f"invalid temporal metadata in TRAIN annotation: {path.name}")
        timestamp = frame_number / float(fps)
        if timestamp < 0 or timestamp > float(duration):
            raise ValueError(f"annotation timestamp exceeds clip duration: {path.name}")
        categories = {annotation["audio_visual_category"] for annotation in annotations}
        asset = {
            "asset_key": f"{video_id}/{frame_number:05d}",
            "video_id": video_id,
            "source_group": group,
            "timestamp_seconds": timestamp,
            "duration_seconds": float(duration),
            "categories": categories,
        }
        assets_by_group[group].append(asset)
        groups.add(group)
    for group in assets_by_group:
        assets_by_group[group].sort(key=lambda item: item["asset_key"])
    return assets_by_group, groups


def _single_category(assets: list[dict]) -> str | None:
    categories = set().union(*(asset["categories"] for asset in assets))
    return next(iter(categories)) if len(categories) == 1 else None


def _choose_asset(assets: list[dict], *rank_parts: str) -> dict:
    return min(assets, key=lambda asset: _rank(*rank_parts, asset["asset_key"]))


def _build_split(
    *,
    split_name: str,
    groups: set[str],
    assets_by_group: dict[str, list[dict]],
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    group_categories = {
        group: _single_category(assets_by_group[group])
        for group in groups
        if group in assets_by_group
    }
    category_groups: dict[str, list[str]] = defaultdict(list)
    for group, category in group_categories.items():
        if category is not None:
            category_groups[category].append(group)
    for category in category_groups:
        category_groups[category].sort(key=lambda group: _rank(split_name, "group", group))

    inputs: list[dict] = []
    sidecars: list[dict] = []
    skipped_ambiguous = 0
    skipped_no_positive = 0
    skipped_no_negative_categories = 0
    target_positions: Counter[int] = Counter()

    for query_group in sorted(groups, key=lambda group: _rank(split_name, "query", group)):
        category = group_categories.get(query_group)
        if category is None:
            skipped_ambiguous += 1
            continue
        positive_groups = [
            candidate for candidate in category_groups[category] if candidate != query_group
        ]
        if not positive_groups:
            skipped_no_positive += 1
            continue
        negative_categories = [
            other
            for other, candidates in category_groups.items()
            if other != category and candidates
        ]
        negative_categories.sort(
            key=lambda other: _rank(split_name, "negative-category", query_group, other)
        )
        if len(negative_categories) < 3:
            skipped_no_negative_categories += 1
            continue

        positive_group = min(
            positive_groups,
            key=lambda candidate: _rank(split_name, "positive-group", query_group, candidate),
        )
        selected_groups = [positive_group]
        selected_categories = [category]
        for negative_category in negative_categories[:3]:
            candidate_group = min(
                category_groups[negative_category],
                key=lambda candidate: _rank(
                    split_name, "negative-group", query_group, negative_category, candidate
                ),
            )
            selected_groups.append(candidate_group)
            selected_categories.append(negative_category)

        query_asset = _choose_asset(
            assets_by_group[query_group], split_name, "query-asset", query_group
        )
        positive_asset = _choose_asset(
            assets_by_group[positive_group],
            split_name,
            "candidate-asset",
            query_group,
            positive_group,
        )
        negative_rows = [
            (
                group,
                candidate_category,
                _choose_asset(
                    assets_by_group[group], split_name, "candidate-asset", query_group, group
                ),
            )
            for group, candidate_category in zip(
                selected_groups[1:], selected_categories[1:], strict=True
            )
        ]
        negative_rows.sort(
            key=lambda item: _rank(split_name, "negative-option-order", query_group, item[0])
        )
        target_index = len(inputs) % 4
        option_rows = list(negative_rows)
        option_rows.insert(target_index, (positive_group, category, positive_asset))
        option_groups = [item[0] for item in option_rows]
        option_categories = [item[1] for item in option_rows]
        candidate_assets = [item[2] for item in option_rows]
        example_id = hashlib.sha256(
            f"avatar-multientity-cross-source-v1:{SEED}:{split_name}:{query_asset['asset_key']}".encode()
        ).hexdigest()
        left_padding = max(0.0, WINDOW_SECONDS / 2 - query_asset["timestamp_seconds"])
        right_padding = max(
            0.0,
            query_asset["timestamp_seconds"]
            + WINDOW_SECONDS / 2
            - query_asset["duration_seconds"],
        )
        inputs.append(
            {
                "example_id": example_id,
                "split": split_name,
                "question": "Which video option shows the source associated with this audio?",
                "audio_asset_key": query_asset["asset_key"],
                "audio_window_center_seconds": query_asset["timestamp_seconds"],
                "audio_window_duration_seconds": WINDOW_SECONDS,
                "audio_window_left_padding_seconds": left_padding,
                "audio_window_right_padding_seconds": right_padding,
                "options": [
                    {
                        "option_id": f"option_{index}",
                        "video_asset_key": candidate_assets[index]["asset_key"],
                        "video_timestamp_seconds": candidate_assets[index]["timestamp_seconds"],
                    }
                    for index in range(4)
                ],
                "target_option_id": f"option_{target_index}",
            }
        )
        sidecars.append(
            {
                "example_id": example_id,
                "query_source_group": query_group,
                "query_category": category,
                "query_asset_key": query_asset["asset_key"],
                "target_option_index": target_index,
                "option_source_groups": option_groups,
                "option_categories": option_categories,
                "option_asset_keys": [asset["asset_key"] for asset in candidate_assets],
            }
        )
        target_positions[target_index] += 1

    summary = {
        "source_group_count": len(groups),
        "single_category_group_count": sum(
            value is not None for value in group_categories.values()
        ),
        "query_count": len(inputs),
        "target_category_count": len({row["query_category"] for row in sidecars}),
        "target_option_position_counts": {
            str(index): target_positions[index] for index in range(4)
        },
        "skipped_ambiguous_source_groups": skipped_ambiguous,
        "skipped_without_same_category_positive": skipped_no_positive,
        "skipped_without_three_negative_categories": skipped_no_negative_categories,
    }
    return inputs, sidecars, summary


def build_corpus(
    *, train_annotations: Path, artifact_root: Path, output_dir: Path
) -> dict[str, Any]:
    train_annotations = train_annotations.resolve()
    if train_annotations.name.lower() != "train":
        raise ValueError("train_annotations must be the isolated TRAIN directory")
    if any(part.lower() == "audit" for part in train_annotations.parts):
        raise ValueError("audit paths are forbidden")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    fit_groups, evaluation_groups, exposure_hashes, _ = collect_prior_group_usage(artifact_root)
    prior_exposed = fit_groups | evaluation_groups
    assets_by_group, all_multientity_groups = _load_multientity_assets(train_annotations)
    candidate_groups = all_multientity_groups - prior_exposed
    train_groups, validation_groups = provisional_group_split(candidate_groups)
    train_inputs, train_sidecars, train_summary = _build_split(
        split_name="train", groups=train_groups, assets_by_group=assets_by_group
    )
    validation_inputs, validation_sidecars, validation_summary = _build_split(
        split_name="validation", groups=validation_groups, assets_by_group=assets_by_group
    )

    for left, right in (
        (train_groups, validation_groups),
        (
            {sidecar["query_source_group"] for sidecar in train_sidecars}
            | {group for sidecar in train_sidecars for group in sidecar["option_source_groups"]},
            {sidecar["query_source_group"] for sidecar in validation_sidecars}
            | {
                group
                for sidecar in validation_sidecars
                for group in sidecar["option_source_groups"]
            },
        ),
        (
            {sidecar["query_asset_key"] for sidecar in train_sidecars}
            | {asset for sidecar in train_sidecars for asset in sidecar["option_asset_keys"]},
            {sidecar["query_asset_key"] for sidecar in validation_sidecars}
            | {asset for sidecar in validation_sidecars for asset in sidecar["option_asset_keys"]},
        ),
    ):
        if left & right:
            raise ValueError("AVATAR source-group or asset overlap across generated splits")
    if not train_inputs or not validation_inputs:
        raise ValueError("both train and validation must contain generated choices")

    train_positions = Counter(
        int(row["target_option_id"].rsplit("_", 1)[1]) for row in train_inputs
    )
    validation_positions = Counter(
        int(row["target_option_id"].rsplit("_", 1)[1]) for row in validation_inputs
    )
    train_position_prior = min(
        range(4), key=lambda index: (-train_positions[index], index)
    )
    position_prior_baseline = {
        "selected_position_from_train_only": train_position_prior,
        "train_accuracy": train_positions[train_position_prior] / len(train_inputs),
        "validation_accuracy": validation_positions[train_position_prior]
        / len(validation_inputs),
        "uniform_chance_accuracy": 0.25,
        "validation_position_counts": {
            str(index): validation_positions[index] for index in range(4)
        },
        "interpretation": "option-position-only baseline; no feature/model fitting",
    }

    output_dir.mkdir(parents=True)
    outputs = {
        "train_model_input.jsonl": train_inputs,
        "train_analysis_sidecar.jsonl": train_sidecars,
        "validation_model_input.jsonl": validation_inputs,
        "validation_analysis_sidecar.jsonl": validation_sidecars,
    }
    for name, rows in outputs.items():
        path = output_dir / name
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    train_tree_hash, train_json_count = _tree_hash(train_annotations)
    manifest = {
        "format": "avatar_multientity_cross_source_choice_corpus_v1",
        "status": "METADATA_FROZEN_PENDING_MEDIA_PREFLIGHT",
        "seed": SEED,
        "dataset_revision": "a7f0146ff704799d044133e61137836cacc1c233",
        "task": {
            "name": "cross-source AV category match over AVATAR Multi-Entity TRAIN groups",
            "question": "Which video option shows the source associated with this audio?",
            "choice_count": 4,
            "query_source_group_category_rule": (
                "one unique Multi-Entity category per query source group"
            ),
            "positive_rule": (
                "one video-frame asset with the same annotated category "
                "from another source group"
            ),
            "negative_rule": "three different categories and three distinct source groups",
            "option_category_labels_in_model_input": False,
            "asset_keys_are_loader_lookup_only": True,
            "timestamps_are_encoder_alignment_metadata_only": True,
            "question_is_constant_and_not_a_scorer_feature": True,
            "scorer_feature_contract": (
                "frozen audio/video feature tensors only; IDs, timestamps, "
                "target labels, and sidecars are not head features"
            ),
        },
        "split": {
            "method": "SHA-256 rank of prior-unexposed TRAIN source-group identities",
            "hash_prefix": "avatar-multientity-dev-v1:17:",
            "train_fraction": 0.8,
            "validation_fraction": 0.2,
            "source_group_overlap": 0,
            "query_and_candidate_source_group_cross_split_overlap": 0,
            "query_and_candidate_asset_cross_split_overlap": 0,
            "sealed_audit_accessed": False,
            "official_validation_annotations_or_media_accessed": False,
        },
        "source": {
            "train_annotation_json_count": train_json_count,
            "train_annotation_tree_sha256": train_tree_hash,
            "prior_fit_or_evaluation_group_set_sha256": _sha256_ids(prior_exposed),
            "prior_exposure_inputs_sha256": exposure_hashes,
            "candidate_multientity_source_group_count": len(candidate_groups),
            "candidate_source_group_set_sha256": _sha256_ids(candidate_groups),
        },
        "counts": {"train": train_summary, "validation": validation_summary},
        "metadata_only_controls": {
            "option_position_prior": position_prior_baseline,
            "source_group_or_category_sidecars_exposed_to_scorer": False,
        },
        "files": {
            name: {
                "rows": len(rows),
                "sha256": _sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name, rows in outputs.items()
        },
        "integrity": {
            "query_source_group_is_not_candidate": True,
            "candidate_source_groups_unique_within_example": True,
            "positive_category_matches_query_category": True,
            "negative_categories_are_distinct": True,
            "source_group_ids_and_labels_are_in_external_sidecars_only": True,
            "media_bytes_read": False,
            "feature_extraction_started": False,
            "model_loaded_or_trained": False,
        },
        "limitations": [
            "This is a derived AV category-match task, not native source localization.",
            "Group identity separation does not verify perceptual media duplicates.",
            "VideoPrism pretraining overlap with AVATAR source videos is unknown.",
            "Validation examples are a development split from AVATAR TRAIN, not a blind audit.",
            "Audio/video stream availability and annotation-time alignment "
            "remain unverified for selected clips.",
        ],
    }
    manifest_path = output_dir / "corpus-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_corpus(
        train_annotations=args.train_annotations,
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "train_queries": manifest["counts"]["train"]["query_count"],
                "validation_queries": manifest["counts"]["validation"]["query_count"],
                "group_overlap": manifest["split"]["source_group_overlap"],
                "sealed_audit_accessed": manifest["split"]["sealed_audit_accessed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
