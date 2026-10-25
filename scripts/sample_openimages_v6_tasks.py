"""Create a redacted deterministic pilot task manifest for one Open Images split."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.audit_openimages_v6_train_support import (
    load_display_names,
    load_image_allowlist,
    sha256_file,
)
from tiny_omni_decision.openimages_options import (
    build_sibling_groups,
    build_split_task_manifest,
    group_verification_labels,
)

SPLIT_REVISIONS = {
    "train": {
        "image_metadata_generation": "1526567626302755",
        "label_generation": "1580742923196487",
    },
    "validation": {
        "image_metadata_generation": "1526567618256414",
        "label_generation": "1557303846177611",
    },
}
HIERARCHY_GENERATION = "1528191513246771"
CC_BY_2_URL = "https://creativecommons.org/licenses/by/2.0/"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def build_manifest_from_files(
    *,
    split: str,
    image_metadata_path: Path,
    labels_path: Path,
    hierarchy_path: Path,
    class_descriptions_path: Path,
    frozen_train_scope_path: Path,
    per_target_limit: int,
    seed: int,
) -> dict[str, Any]:
    if split not in SPLIT_REVISIONS:
        raise ValueError("split must be train or validation")
    scope = json.loads(frozen_train_scope_path.read_text(encoding="utf-8"))
    if scope.get("purpose") != "train-only Open Images V6 hierarchy-choice support audit":
        raise ValueError("target scope is not a recognized train-only support report")
    target_labels = tuple(
        item["label_id"] for item in scope.get("selected_target_classes", [])
    )
    if not target_labels:
        raise ValueError("frozen training target scope is empty")

    allowed_image_ids, metadata_counts = load_image_allowlist(image_metadata_path, split)
    display_names = load_display_names(class_descriptions_path)
    with hierarchy_path.open("r", encoding="utf-8") as stream:
        sibling_groups = build_sibling_groups(json.load(stream), minimum_alternatives=3)
    with labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        examples = build_split_task_manifest(
            grouped_labels=group_verification_labels(reader),
            allowed_image_ids=allowed_image_ids,
            target_labels=target_labels,
            sibling_groups=sibling_groups,
            display_names=display_names,
            split=split,
            per_target_limit=per_target_limit,
            seed=seed,
        )

    source_hashes = {
        "image_metadata_sha256": sha256_file(image_metadata_path),
        "human_labels_sha256": sha256_file(labels_path),
        "hierarchy_sha256": sha256_file(hierarchy_path),
        "class_descriptions_sha256": sha256_file(class_descriptions_path),
        "frozen_train_scope_sha256": sha256_file(frozen_train_scope_path),
    }
    revision_payload = json.dumps(
        {
            "split": split,
            "generations": SPLIT_REVISIONS[split],
            "hierarchy_generation": HIERARCHY_GENERATION,
            "frozen_target_labels": sorted(target_labels),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        **examples,
        "source_revision": hashlib.sha1(revision_payload.encode("utf-8")).hexdigest(),
        "source_generations": {
            **SPLIT_REVISIONS[split],
            "hierarchy_generation": HIERARCHY_GENERATION,
        },
        "source_hashes": source_hashes,
        "image_metadata_counts": metadata_counts,
        "source_license_status": (
            "metadata lists CC BY 2.0; individual landing-page verification remains required"
        ),
        "individual_rights_verified": False,
        "model_loaded": False,
    }


def extract_private_attribution_rows(
    path: Path, *, expected_split: str, selected_image_ids: set[str]
) -> list[dict[str, Any]]:
    """Extract rights-relevant fields for selected IDs; caller stores them privately."""
    if expected_split not in SPLIT_REVISIONS:
        raise ValueError("expected_split must be train or validation")
    if not selected_image_ids:
        raise ValueError("selected_image_ids must not be empty")
    required = {
        "ImageID",
        "Subset",
        "OriginalURL",
        "OriginalLandingURL",
        "License",
        "AuthorProfileURL",
        "Author",
        "OriginalMD5",
    }
    receipts: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("image metadata is missing attribution receipt columns")
        for row in reader:
            image_id = row["ImageID"]
            if image_id not in selected_image_ids:
                continue
            if image_id in receipts:
                raise ValueError(f"duplicate selected ImageID in metadata: {image_id}")
            if row["Subset"] != expected_split:
                raise ValueError(f"selected image is in wrong split: {image_id}")
            if row["License"].rstrip("/") != CC_BY_2_URL.rstrip("/"):
                raise ValueError(f"selected image lacks CC BY 2.0 metadata: {image_id}")
            for field in required - {"ImageID", "Subset"}:
                if not row[field]:
                    raise ValueError(f"selected image is missing {field}: {image_id}")
            thumbnail_url = row.get("Thumbnail300KURL", "")
            try:
                decoded_md5 = base64.b64decode(row["OriginalMD5"], validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"selected image has invalid OriginalMD5: {image_id}") from exc
            if len(decoded_md5) != 16:
                raise ValueError(f"selected image has invalid MD5 length: {image_id}")
            if row["OriginalMD5"] in source_hashes:
                raise ValueError("duplicate OriginalMD5 within selected split sample")
            source_hashes[row["OriginalMD5"]] = image_id
            receipts[image_id] = {
                field: row[field]
                for field in (
                    "ImageID",
                    "Subset",
                    "OriginalURL",
                    "OriginalLandingURL",
                    "License",
                    "AuthorProfileURL",
                    "Author",
                    "OriginalMD5",
                )
            } | {
                "Thumbnail300KURL": thumbnail_url or None,
                "preferred_media_url": thumbnail_url or row["OriginalURL"],
                "media_url_type": "thumbnail_300k" if thumbnail_url else "original",
                "landing_page_check_status": "not_checked",
                "individual_rights_verified": False,
            }
    missing = selected_image_ids - receipts.keys()
    if missing:
        raise ValueError(f"selected images missing from metadata: {len(missing)}")
    return [receipts[image_id] for image_id in sorted(receipts)]


def require_disjoint_attribution_rows(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, int]:
    """Fail if IDs or source MD5s repeat within or across the selected splits."""
    train_ids = [row.get("ImageID") for row in train_rows]
    validation_ids = [row.get("ImageID") for row in validation_rows]
    if any(not isinstance(item, str) or not item for item in (*train_ids, *validation_ids)):
        raise ValueError("attribution rows need nonempty ImageID values")
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("duplicate ImageID within a selected split sample")
    id_overlap = set(train_ids) & set(validation_ids)
    if id_overlap:
        raise ValueError(f"ImageID overlap across splits: {len(id_overlap)}")
    train_hashes = [row.get("OriginalMD5") for row in train_rows]
    validation_hashes = [row.get("OriginalMD5") for row in validation_rows]
    if any(not isinstance(item, str) or not item for item in (*train_hashes, *validation_hashes)):
        raise ValueError("attribution rows need nonempty OriginalMD5 values")
    if len(set(train_hashes)) != len(train_hashes) or len(set(validation_hashes)) != len(
        validation_hashes
    ):
        raise ValueError("duplicate OriginalMD5 within a selected split sample")
    md5_overlap = set(train_hashes) & set(validation_hashes)
    if md5_overlap:
        raise ValueError(f"OriginalMD5 overlap across splits: {len(md5_overlap)}")
    return {"train_rows": len(train_rows), "validation_rows": len(validation_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLIT_REVISIONS), required=True)
    parser.add_argument("--image-metadata", type=Path, required=True)
    parser.add_argument("--human-labels", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--class-descriptions", type=Path, required=True)
    parser.add_argument("--frozen-train-scope", type=Path, required=True)
    parser.add_argument("--per-target-limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-attribution-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    private_output = args.private_attribution_output.resolve()
    if private_output.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("private attribution output must be outside the repository")
    if private_output.exists():
        raise FileExistsError(f"refusing to overwrite private receipt {private_output}")
    manifest = build_manifest_from_files(
        split=args.split,
        image_metadata_path=args.image_metadata,
        labels_path=args.human_labels,
        hierarchy_path=args.hierarchy,
        class_descriptions_path=args.class_descriptions,
        frozen_train_scope_path=args.frozen_train_scope,
        per_target_limit=args.per_target_limit,
        seed=args.seed,
    )
    receipt_rows = extract_private_attribution_rows(
        args.image_metadata,
        expected_split=args.split,
        selected_image_ids={item["image_id"] for item in manifest["examples"]},
    )
    private_output.parent.mkdir(parents=True, exist_ok=True)
    private_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in receipt_rows
    )
    private_output.write_text(private_payload, encoding="utf-8")
    manifest["private_attribution_receipt_rows"] = len(receipt_rows)
    manifest["private_attribution_receipt_sha256"] = hashlib.sha256(
        private_payload.encode("utf-8")
    ).hexdigest()
    manifest["private_attribution_receipt_outside_repository"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "split": manifest["split"],
                "example_count": manifest["example_count"],
                "counts_by_target": manifest["counts_by_target"],
                "source_revision": manifest["source_revision"],
                "private_attribution_receipt_rows": len(receipt_rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
