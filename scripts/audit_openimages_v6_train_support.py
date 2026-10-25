"""Audit train-only support for Open Images V6 hierarchy-matched choices.

This utility reads image metadata and human-verified image-level labels. It
does not access validation, official test, sealed audit, or image media.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from tiny_omni_decision.openimages_options import (
    EXCLUDED_OPENIMAGES_NAME_TERMS,
    MINIMUM_OPENIMAGES_TRAIN_SUPPORT,
    build_sibling_groups,
    group_verification_labels,
    iter_target_candidates,
    select_target_classes,
)

CC_BY_2_URL = "https://creativecommons.org/licenses/by/2.0/"


def load_image_allowlist(path: Path, expected_split: str) -> tuple[set[str], dict[str, int]]:
    """Return CC BY 2.0 IDs for one source split without retaining metadata."""
    if expected_split not in {"train", "validation"}:
        raise ValueError("expected_split must be train or validation")
    allowed: set[str] = set()
    seen: set[str] = set()
    rows = 0
    split_rows = 0
    cc_by_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ImageID", "Subset", "License"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("image metadata is missing required columns")
        for row in reader:
            rows += 1
            image_id = row["ImageID"]
            if not image_id:
                raise ValueError("image metadata contains an empty ImageID")
            if image_id in seen:
                raise ValueError(f"duplicate ImageID in image metadata: {image_id}")
            seen.add(image_id)
            if row["Subset"] != expected_split:
                continue
            split_rows += 1
            if row["License"].rstrip("/") == CC_BY_2_URL.rstrip("/"):
                allowed.add(image_id)
                cc_by_rows += 1
    return allowed, {
        "rows": rows,
        f"{expected_split}_rows": split_rows,
        "cc_by_2_rows": cc_by_rows,
    }


def load_train_image_allowlist(path: Path) -> tuple[set[str], dict[str, int]]:
    """Return CC BY 2.0 train IDs and coarse counts without retaining metadata."""
    return load_image_allowlist(path, "train")


def load_display_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"LabelName", "DisplayName"}.issubset(reader.fieldnames or ()):
            raise ValueError("class descriptions are missing required columns")
        for row in reader:
            label = row["LabelName"]
            display_name = row["DisplayName"]
            if not label or not display_name:
                raise ValueError("class description contains an empty field")
            if label in names:
                raise ValueError(f"duplicate class label description: {label}")
            names[label] = display_name
    return names


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_train_support(
    *,
    train_images: Path,
    train_labels: Path,
    hierarchy_path: Path,
    class_descriptions: Path,
) -> dict[str, Any]:
    """Produce a train-only class support summary with source-file hashes."""
    allowed_ids, metadata_counts = load_train_image_allowlist(train_images)
    display_names = load_display_names(class_descriptions)
    with hierarchy_path.open("r", encoding="utf-8") as stream:
        hierarchy = json.load(stream)
    sibling_groups = build_sibling_groups(hierarchy, minimum_alternatives=3)

    counts: dict[str, int] = {}
    supported_image_ids: set[str] = set()
    with train_labels.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"ImageID", "Source", "LabelName", "Confidence"}.issubset(
            reader.fieldnames or ()
        ):
            raise ValueError("train labels are missing required columns")
        groups = group_verification_labels(reader)
        for candidate in iter_target_candidates(
            _licensed_groups(groups, allowed_ids),
            target_labels=tuple(sorted(sibling_groups)),
            sibling_groups=sibling_groups,
        ):
            target = candidate["target_label"]
            counts[target] = counts.get(target, 0) + 1
            supported_image_ids.add(candidate["image_id"])

    selected = select_target_classes(
        counts,
        display_names,
        minimum_train_support=MINIMUM_OPENIMAGES_TRAIN_SUPPORT,
        excluded_name_terms=EXCLUDED_OPENIMAGES_NAME_TERMS,
    )
    support_rows = [
        {
            "label_id": label,
            "display_name": display_names.get(label, ""),
            "unique_train_images": count,
        }
        for label, count in sorted(counts.items())
    ]
    selected_rows = [
        {
            "label_id": label,
            "display_name": display_names[label],
            "unique_train_images": count,
        }
        for label, count in sorted(selected.items())
    ]
    return {
        "schema_version": 1,
        "purpose": "train-only Open Images V6 hierarchy-choice support audit",
        "media_downloaded": False,
        "validation_or_test_accessed": False,
        "metadata": {
            **metadata_counts,
            "cc_by_2_train_ids_allowlisted": len(allowed_ids),
            "sha256": sha256_file(train_images),
        },
        "train_labels_sha256": sha256_file(train_labels),
        "hierarchy_sha256": sha256_file(hierarchy_path),
        "class_descriptions_sha256": sha256_file(class_descriptions),
        "label_policy": {
            "source": "verification",
            "confidence_values": [0.0, 1.0],
            "negative_policy": "explicit negative sibling labels only",
            "minimum_explicit_negative_siblings": 3,
        },
        "target_scope_policy": {
            "minimum_unique_train_support": MINIMUM_OPENIMAGES_TRAIN_SUPPORT,
            "excluded_display_name_terms": sorted(EXCLUDED_OPENIMAGES_NAME_TERMS),
        },
        "hierarchy_sibling_target_class_count": len(counts),
        "images_supporting_any_target": len(supported_image_ids),
        "total_positive_target_options": sum(counts.values()),
        "selected_target_class_count": len(selected_rows),
        "selected_target_classes": selected_rows,
        "train_support_by_class": support_rows,
    }


def _licensed_groups(groups, allowed_ids):
    for group in groups:
        if group["image_id"] in allowed_ids:
            yield group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--class-descriptions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = audit_train_support(
        train_images=args.train_images,
        train_labels=args.train_labels,
        hierarchy_path=args.hierarchy,
        class_descriptions=args.class_descriptions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_target_class_count": result["selected_target_class_count"],
                "images_supporting_any_target": result["images_supporting_any_target"],
                "total_positive_target_options": result["total_positive_target_options"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
