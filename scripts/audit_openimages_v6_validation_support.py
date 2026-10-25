"""Check validation support for classes frozen by the train-only audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.audit_openimages_v6_train_support import (
    load_image_allowlist,
    sha256_file,
)
from tiny_omni_decision.openimages_options import (
    build_sibling_groups,
    group_verification_labels,
    iter_target_candidates,
)


def audit_validation_support(
    *,
    validation_images: Path,
    validation_labels: Path,
    hierarchy_path: Path,
    frozen_target_labels: tuple[str, ...],
) -> dict[str, Any]:
    """Count validation support without selecting or dropping target classes."""
    if not frozen_target_labels or len(set(frozen_target_labels)) != len(frozen_target_labels):
        raise ValueError("frozen_target_labels must be nonempty and unique")
    allowlist, metadata_counts = load_image_allowlist(validation_images, "validation")
    with hierarchy_path.open("r", encoding="utf-8") as stream:
        sibling_groups = build_sibling_groups(json.load(stream), minimum_alternatives=3)
    missing = set(frozen_target_labels) - set(sibling_groups)
    if missing:
        raise ValueError(f"frozen targets lack hierarchy siblings: {sorted(missing)}")

    support = {label: 0 for label in sorted(frozen_target_labels)}
    task_image_ids: set[str] = set()
    with validation_labels.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"ImageID", "Source", "LabelName", "Confidence"}.issubset(
            reader.fieldnames or ()
        ):
            raise ValueError("validation labels are missing required columns")
        for group in group_verification_labels(reader):
            if group["image_id"] not in allowlist:
                continue
            for candidate in iter_target_candidates(
                [group],
                target_labels=frozen_target_labels,
                sibling_groups=sibling_groups,
            ):
                support[candidate["target_label"]] += 1
                task_image_ids.add(candidate["image_id"])

    target_payload = json.dumps(sorted(frozen_target_labels), separators=(",", ":"))
    return {
        "schema_version": 1,
        "purpose": "validation feasibility audit for train-frozen Open Images targets",
        "target_selection_source": "training-only frozen manifest",
        "frozen_target_labels": sorted(frozen_target_labels),
        "frozen_target_labels_sha256": hashlib.sha256(target_payload.encode()).hexdigest(),
        "media_downloaded": False,
        "model_or_checkpoint_loaded": False,
        "metadata": {**metadata_counts, "sha256": sha256_file(validation_images)},
        "validation_labels_sha256": sha256_file(validation_labels),
        "hierarchy_sha256": sha256_file(hierarchy_path),
        "unique_validation_images_with_task": len(task_image_ids),
        "support_by_target": support,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-images", type=Path, required=True)
    parser.add_argument("--validation-labels", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--train-support-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_report = json.loads(args.train_support_report.read_text(encoding="utf-8"))
    target_labels = tuple(
        item["label_id"] for item in train_report["selected_target_classes"]
    )
    result = audit_validation_support(
        validation_images=args.validation_images,
        validation_labels=args.validation_labels,
        hierarchy_path=args.hierarchy,
        frozen_target_labels=target_labels,
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
                "frozen_target_count": len(target_labels),
                "unique_validation_images_with_task": result[
                    "unique_validation_images_with_task"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
