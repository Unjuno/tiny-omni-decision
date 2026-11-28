"""Verify identity and normalized-content disjointness for Physion++ train/dev splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tiny_omni_decision.dataset import check_train_eval_splits
from tiny_omni_decision.schema import DecisionExample


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split(
    path: Path, expected_split: str
) -> tuple[list[dict[str, Any]], list[DecisionExample]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{expected_split} manifest is empty")
    if any(row.get("split") != expected_split for row in rows):
        raise ValueError(f"manifest contains rows outside the {expected_split} split")

    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{expected_split} sample IDs must be unique non-empty strings")
    source_ids = [row.get("source_record_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in source_ids) or len(
        set(source_ids)
    ) != len(source_ids):
        raise ValueError(f"{expected_split} source record IDs must be unique non-empty strings")
    for row in rows:
        if row.get("source") != "physionpp/readout":
            raise ValueError(f"unexpected source in {expected_split}: {row.get('source')!r}")
        if not isinstance(row.get("source_asset_group_id"), str) or not row[
            "source_asset_group_id"
        ]:
            raise ValueError(f"missing source asset group in {expected_split}: {row['id']}")
        media = row.get("media")
        if (
            not isinstance(media, list)
            or len(media) != 1
            or media[0].get("kind") != "video"
            or not isinstance(media[0].get("sha256"), str)
            or not isinstance(media[0].get("path"), str)
        ):
            raise ValueError(f"expected one SHA-pinned video in {expected_split}: {row['id']}")
    for key in ("sha256", "path"):
        references = [row["media"][0][key] for row in rows]
        if len(set(references)) != len(references):
            raise ValueError(f"{expected_split} video {key} values must be unique")

    examples = [DecisionExample.model_validate(row) for row in rows]
    return rows, examples


def _identity_overlap(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], key: str
) -> dict[str, int]:
    def values(rows: list[dict[str, Any]]) -> set[str]:
        return {row[key] for row in rows}

    train_values = values(train_rows)
    validation_values = values(validation_rows)
    return {
        "train_unique": len(train_values),
        "validation_unique": len(validation_values),
        "shared": len(train_values & validation_values),
    }


def _media_overlap(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], key: str
) -> dict[str, int]:
    def values(rows: list[dict[str, Any]]) -> set[str]:
        return {row["media"][0][key] for row in rows}

    train_values = values(train_rows)
    validation_values = values(validation_rows)
    return {
        "train_unique": len(train_values),
        "validation_unique": len(validation_values),
        "shared": len(train_values & validation_values),
    }


def audit_train_validation_splits(train_path: Path, validation_path: Path) -> dict[str, Any]:
    """Check record, scene, media, and normalized-content separation without decoding media."""
    train_rows, train_examples = _load_split(train_path, "train")
    validation_rows, validation_examples = _load_split(validation_path, "validation")
    if len({row["source_revision"] for row in train_rows + validation_rows}) != 1:
        raise ValueError("train and validation must use one pinned source revision")

    overlaps = {
        "record_id": _identity_overlap(train_rows, validation_rows, "id"),
        "source_record_id": _identity_overlap(
            train_rows, validation_rows, "source_record_id"
        ),
        "source_asset_group_id": _identity_overlap(
            train_rows, validation_rows, "source_asset_group_id"
        ),
        "media_sha256": _media_overlap(train_rows, validation_rows, "sha256"),
        "media_path": _media_overlap(train_rows, validation_rows, "path"),
        "normalized_content_and_source_id": check_train_eval_splits(
            train_examples, validation_examples
        ),
    }
    shared_counts = {
        name: result["shared"]
        for name, result in overlaps.items()
        if name != "normalized_content_and_source_id"
    }
    if any(shared_counts.values()):
        raise ValueError(f"train/validation identity overlap: {shared_counts}")

    return {
        "status": "disjoint",
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "source_revision": train_rows[0]["source_revision"],
        "train_manifest_sha256": _sha256_file(train_path),
        "validation_manifest_sha256": _sha256_file(validation_path),
        "overlaps": overlaps,
        "media_decoded": False,
        "model_loaded": False,
        "model_metrics_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_train_validation_splits(args.train, args.validation)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
