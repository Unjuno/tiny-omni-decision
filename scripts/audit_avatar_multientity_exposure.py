"""Summarize prior model exposure of AVATAR TRAIN Multi-Entity source groups.

This audit reads only the explicitly supplied AVATAR TRAIN annotation tree and
named prior fit/evaluation artifacts. It never opens official validation or
sealed-audit annotations/media, and it emits aggregate counts and hashes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CLIP_ID = re.compile(r"([A-Za-z0-9_-]{11})_(\d+)\Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_ids(values: set[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_group_from_annotation(path: Path, row: dict[str, Any]) -> str:
    match = CLIP_ID.fullmatch(path.parent.name)
    if match is None:
        raise ValueError(f"invalid AVATAR clip directory: {path.parent.name}")
    source_group, offset = match.groups()
    if row.get("video_id") != f"{source_group}_{int(offset)}":
        raise ValueError(f"video_id does not match clip directory: {path.name}")
    return source_group


def _read_sidecar(path: Path, destination: set[str]) -> int:
    rows = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows += 1
            for key in ("audio_source_group", "target_source_group", "source_group"):
                value = row.get(key)
                if value:
                    destination.add(value)
            destination.update(row.get("option_source_groups") or [])
    return rows


def _sidecar_for(directory: Path, split: str, *, allow_plain_jsonl: bool = False) -> Path:
    names = [f"{split}-analysis-sidecar.jsonl", f"{split}-sidecar.jsonl"]
    if allow_plain_jsonl:
        names.append(f"{split}.jsonl")
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing inventoried {split} sidecar in {directory}")


def provisional_group_split(groups: set[str]) -> tuple[set[str], set[str]]:
    """Return a deterministic illustrative 80/20 split; this does not freeze data."""
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"avatar-multientity-dev-v1:17:{group}".encode()
        ).hexdigest(),
    )
    validation_count = round(len(ranked) * 0.2)
    validation = set(ranked[:validation_count])
    return set(ranked[validation_count:]), validation


def collect_prior_group_usage(
    artifact_root: Path,
) -> tuple[set[str], set[str], dict[str, str], dict[str, int]]:
    """Return source groups and hashes from the inventoried prior AVATAR runs."""
    fit_groups: set[str] = set()
    evaluation_groups: set[str] = set()
    input_hashes: dict[str, str] = {}
    sidecar_rows: dict[str, int] = {}

    choice_dirs = [
        artifact_root / "avatar-av-decision-20261101-attempt-01" / f"option-choices-{i:02d}"
        for i in range(1, 5)
    ]
    choice_dirs.append(
        artifact_root
        / "avatar-av-decision-20261102-attempt-01"
        / "cross-source-option-choices-02"
    )
    for directory in choice_dirs:
        for split, destination in (("train", fit_groups), ("validation", evaluation_groups)):
            path = _sidecar_for(directory, split)
            label = f"{directory.name}/{path.name}"
            sidecar_rows[label] = _read_sidecar(path, destination)
            input_hashes[label] = _sha256_file(path)

    category_dir = artifact_root / "avatar-category-transfer-20261102-attempt-03"
    split_manifest_path = category_dir / "split-manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    input_hashes["avatar-category-transfer/split-manifest.json"] = _sha256_file(
        split_manifest_path
    )
    for name, destination in (
        ("fit", fit_groups),
        ("inner_validation", evaluation_groups),
        ("held_category_test", evaluation_groups),
    ):
        destination.update(
            split_manifest.get("splits", {}).get(name, {}).get("query_groups", [])
        )
    held_predictions = category_dir / "held-category-predictions.jsonl"
    sidecar_rows["avatar-category-transfer/held-category-predictions.jsonl"] = (
        _read_sidecar(held_predictions, evaluation_groups)
    )
    input_hashes["avatar-category-transfer/held-category-predictions.jsonl"] = (
        _sha256_file(held_predictions)
    )

    for directory in (
        artifact_root / "avatar-dinov2-train-fold-20261102-attempt-01",
        artifact_root / "avatar-native-localization-20261102-attempt-01" / "fold",
    ):
        for split, destination in (("train", fit_groups), ("validation", evaluation_groups)):
            path = _sidecar_for(directory, split, allow_plain_jsonl=True)
            label = f"{directory.name}/{path.name}"
            sidecar_rows[label] = _read_sidecar(path, destination)
            input_hashes[label] = _sha256_file(path)
    return fit_groups, evaluation_groups, input_hashes, sidecar_rows


def build_report(
    *,
    train_annotations: Path,
    artifact_root: Path,
    extracted_media_dir: Path | None = None,
) -> dict[str, Any]:
    train_annotations = train_annotations.resolve()
    if train_annotations.name.lower() != "train":
        raise ValueError("train_annotations must be the isolated TRAIN directory")
    if any(part.lower() == "audit" for part in train_annotations.parts):
        raise ValueError("audit paths are forbidden")

    fit_groups, evaluation_groups, input_hashes, sidecar_rows = collect_prior_group_usage(
        artifact_root
    )

    all_train_groups: set[str] = set()
    multi_entity_groups: set[str] = set()
    multi_entity_clips: set[str] = set()
    multi_entity_annotations = 0
    multi_entity_bbox_count = 0
    multi_entity_categories: set[str] = set()
    train_file_hash = hashlib.sha256()
    train_json_count = 0

    for path in sorted(train_annotations.rglob("*.json")):
        relative = path.relative_to(train_annotations).as_posix()
        file_hash = _sha256_file(path)
        train_file_hash.update(relative.encode("utf-8") + b"\0")
        train_file_hash.update(file_hash.encode("ascii") + b"\n")
        row = json.loads(path.read_text(encoding="utf-8"))
        group = _source_group_from_annotation(path, row)
        all_train_groups.add(group)
        annotations = [
            annotation
            for annotation in row.get("annotations", [])
            if isinstance(annotation, dict) and annotation.get("task") == "Multi-Entity"
        ]
        if annotations:
            multi_entity_groups.add(group)
            multi_entity_clips.add(row["video_id"])
            multi_entity_annotations += len(annotations)
            for annotation in annotations:
                category = annotation.get("audio_visual_category")
                if category:
                    multi_entity_categories.add(category)
                if isinstance(annotation.get("bbox"), list) and len(annotation["bbox"]) == 4:
                    multi_entity_bbox_count += 1
        train_json_count += 1

    exposed_groups = fit_groups | evaluation_groups
    eligible_groups = multi_entity_groups - exposed_groups
    eligible_clips: set[str] = set()
    eligible_group_categories: dict[str, set[str]] = defaultdict(set)
    eligible_clip_categories: dict[str, set[str]] = defaultdict(set)
    eligible_category_annotations: Counter[str] = Counter()
    eligible_frames_with_multiple_categories = 0
    eligible_annotation_count = 0
    eligible_bbox_count = 0
    # Re-read TRAIN annotations only to count clips for the eligible source groups.
    # No media bytes are opened.
    for path in train_annotations.rglob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        group = _source_group_from_annotation(path, row)
        annotations = [
            annotation
            for annotation in row.get("annotations", [])
            if isinstance(annotation, dict) and annotation.get("task") == "Multi-Entity"
        ]
        if group not in eligible_groups or not annotations:
            continue
        clip_id = row["video_id"]
        eligible_clips.add(clip_id)
        frame_categories = set()
        eligible_annotation_count += len(annotations)
        for annotation in annotations:
            category = annotation.get("audio_visual_category")
            if category:
                frame_categories.add(category)
                eligible_group_categories[group].add(category)
                eligible_clip_categories[clip_id].add(category)
                eligible_category_annotations[category] += 1
            if isinstance(annotation.get("bbox"), list) and len(annotation["bbox"]) == 4:
                eligible_bbox_count += 1
        if len(frame_categories) >= 2:
            eligible_frames_with_multiple_categories += 1

    eligible_media_files = 0
    media_file_count: int | None = None
    if extracted_media_dir is not None and extracted_media_dir.is_dir():
        media_file_count = 0
        for path in extracted_media_dir.iterdir():
            if not path.is_file():
                continue
            media_file_count += 1
            match = CLIP_ID.fullmatch(path.stem)
            if match and match.group(1) in eligible_groups:
                eligible_media_files += 1

    provisional_train_groups, provisional_validation_groups = provisional_group_split(
        eligible_groups
    )
    provisional_category_groups: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    provisional_clip_counts = {"train": 0, "validation": 0}
    for clip_id, _categories in eligible_clip_categories.items():
        match = CLIP_ID.fullmatch(clip_id)
        if match is None:
            raise ValueError(f"invalid eligible Multi-Entity video_id: {clip_id}")
        group = match.group(1)
        split_name = "validation" if group in provisional_validation_groups else "train"
        provisional_clip_counts[split_name] += 1
    for group, categories in eligible_group_categories.items():
        split_name = "validation" if group in provisional_validation_groups else "train"
        provisional_category_groups[split_name].update(categories)
    common_categories = (
        set(provisional_category_groups["train"])
        & set(provisional_category_groups["validation"])
    )

    return {
        "format": "avatar_multientity_exposure_audit_v1",
        "source": {
            "dataset_revision": "a7f0146ff704799d044133e61137836cacc1c233",
            "metadata_archive_sha256": (
                "97e9f09e9f4325ac888db2886da22ca4650eafb8848721230a4e5d80fb6a4183"
            ),
            "train_annotations_sha256": train_file_hash.hexdigest(),
            "train_annotation_json_count": train_json_count,
        },
        "prior_usage": {
            "fit_source_group_count": len(fit_groups),
            "evaluation_source_group_count": len(evaluation_groups),
            "fit_evaluation_source_group_intersection_count": len(fit_groups & evaluation_groups),
            "fit_or_evaluation_source_group_count": len(exposed_groups),
            "fit_source_group_set_sha256": _sha256_ids(fit_groups),
            "evaluation_source_group_set_sha256": _sha256_ids(evaluation_groups),
            "union_source_group_set_sha256": _sha256_ids(exposed_groups),
            "input_file_sha256": input_hashes,
            "sidecar_row_counts": sidecar_rows,
        },
        "train_only_multientity": {
            "source_group_count": len(all_train_groups),
            "multi_entity_source_group_count": len(multi_entity_groups),
            "multi_entity_clip_count": len(multi_entity_clips),
            "multi_entity_annotation_count": multi_entity_annotations,
            "four_value_bbox_annotation_count": multi_entity_bbox_count,
            "category_count": len(multi_entity_categories),
            "fit_exposed_group_count": len(multi_entity_groups & fit_groups),
            "evaluation_exposed_group_count": len(multi_entity_groups & evaluation_groups),
            "fit_and_evaluation_exposed_group_count": len(
                multi_entity_groups & fit_groups & evaluation_groups
            ),
            "distinct_prior_fit_or_evaluation_exposed_group_count": len(
                multi_entity_groups & exposed_groups
            ),
            "unexposed_to_prior_fit_or_evaluation_group_count": len(eligible_groups),
            "unexposed_group_set_sha256": _sha256_ids(eligible_groups),
            "all_train_groups_unexposed_count": len(all_train_groups - exposed_groups),
            "eligible_multi_entity_clip_count": len(eligible_clips),
            "eligible_multi_entity_annotation_count": eligible_annotation_count,
            "eligible_four_value_bbox_annotation_count": eligible_bbox_count,
            "eligible_multi_category_frame_count": eligible_frames_with_multiple_categories,
            "eligible_multi_category_clip_count": sum(
                len(categories) >= 2 for categories in eligible_clip_categories.values()
            ),
            "eligible_multi_category_source_group_count": sum(
                len(categories) >= 2 for categories in eligible_group_categories.values()
            ),
            "eligible_category_count": len(eligible_category_annotations),
            "eligible_categories_with_at_least_20_annotations": sum(
                count >= 20 for count in eligible_category_annotations.values()
            ),
            "eligible_minimum_category_source_group_support": min(
                (
                    sum(category in categories for categories in eligible_group_categories.values())
                    for category in eligible_category_annotations
                ),
                default=0,
            ),
            "provisional_hash_split_not_frozen": {
                "seed": 17,
                "train_source_group_count": len(provisional_train_groups),
                "validation_source_group_count": len(provisional_validation_groups),
                "source_group_overlap_count": len(
                    provisional_train_groups & provisional_validation_groups
                ),
                "train_annotated_clip_count": provisional_clip_counts["train"],
                "validation_annotated_clip_count": provisional_clip_counts["validation"],
                "train_category_count": len(provisional_category_groups["train"]),
                "validation_category_count": len(provisional_category_groups["validation"]),
                "category_count_in_both_splits": len(common_categories),
                "categories_with_four_source_groups_in_both_splits": sum(
                    provisional_category_groups["train"][category] >= 4
                    and provisional_category_groups["validation"][category] >= 4
                    for category in common_categories
                ),
                "minimum_validation_source_groups_per_shared_category": min(
                    (
                        provisional_category_groups["validation"][category]
                        for category in common_categories
                    ),
                    default=0,
                ),
                "task_or_split_frozen": False,
            },
        },
        "existing_media_name_scan": {
            "directory_scanned": extracted_media_dir is not None and media_file_count is not None,
            "file_count": media_file_count,
            "eligible_source_group_media_file_count": eligible_media_files,
            "media_bytes_read": False,
        },
        "sealed_audit_accessed": False,
        "official_validation_annotations_or_media_accessed": False,
        "source_group_ids_emitted": False,
        "interpretation": (
            "Unexposed means absent from inventoried prior fit/evaluation source-group sets. "
            "TRAIN annotations were already extracted and preflighted; "
            "this is not a blind label set."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--extracted-media-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    report = build_report(
        train_annotations=args.train_annotations,
        artifact_root=args.artifact_root,
        extracted_media_dir=args.extracted_media_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        "output": str(args.output),
        "train_source_groups": report["train_only_multientity"]["source_group_count"],
        "multi_entity_groups": report["train_only_multientity"]["multi_entity_source_group_count"],
        "unexposed_multi_entity_groups": report["train_only_multientity"][
            "unexposed_to_prior_fit_or_evaluation_group_count"
        ],
        "sealed_audit_accessed": report["sealed_audit_accessed"],
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
