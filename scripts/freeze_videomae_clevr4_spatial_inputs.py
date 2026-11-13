"""Freeze train and selector media rows for the spatial-grid candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tiny_omni_decision.dataset import sha256_file


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _ordered_id_hash(rows: list[dict[str, Any]]) -> str:
    ids = "\n".join(str(row["id"]) for row in rows) + "\n"
    return hashlib.sha256(ids.encode()).hexdigest()


def combine_train_selector_rows(
    train_rows: list[dict[str, Any]], selector_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    train_ids = [str(row["id"]) for row in train_rows]
    selector_ids = [str(row["id"]) for row in selector_rows]
    train_media = [str(row["media_sha256"]) for row in train_rows]
    selector_media = [str(row["media_sha256"]) for row in selector_rows]
    if len(set(train_ids)) != len(train_ids) or len(set(selector_ids)) != len(selector_ids):
        raise ValueError("duplicate ID inside train or selector manifest")
    if len(set(train_media)) != len(train_media) or len(set(selector_media)) != len(selector_media):
        raise ValueError("duplicate media identity inside train or selector manifest")
    if set(train_ids) & set(selector_ids):
        raise ValueError("train and selector IDs overlap")
    if set(train_media) & set(selector_media):
        raise ValueError("train and selector media identities overlap")
    return train_rows + selector_rows


def freeze(*, inputs: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite spatial inputs {output}")
    receipt_path = inputs / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "FROZEN_VIDEOMAE_CLEVR4_PATH_A_INPUTS":
        raise ValueError("parent frozen input receipt is missing")
    if receipt.get("sealed_audit_loaded") is not False:
        raise ValueError("parent receipt does not protect the sealed audit")
    if receipt.get("cross_split_id_overlap") != 0:
        raise ValueError("parent split has ID overlap")
    if receipt.get("cross_split_media_sha256_overlap") != 0:
        raise ValueError("parent split has media overlap")
    if receipt.get("config_sha256") != sha256_file(inputs / "config.json"):
        raise ValueError("parent frozen config hash mismatch")
    split_files = (
        "train-labels.jsonl",
        "selector-labels.jsonl",
        "train-media-manifest.jsonl",
        "selector-media-manifest.jsonl",
    )
    for name in split_files:
        if sha256_file(inputs / name) != receipt["file_hashes"][name]:
            raise ValueError(f"parent frozen split file hash mismatch: {name}")
    train_labels = _load_jsonl(inputs / "train-labels.jsonl")
    selector_labels = _load_jsonl(inputs / "selector-labels.jsonl")
    train_rows = _load_jsonl(inputs / "train-media-manifest.jsonl")
    selector_rows = _load_jsonl(inputs / "selector-media-manifest.jsonl")
    if [str(row["id"]) for row in train_labels] != [str(row["id"]) for row in train_rows]:
        raise ValueError("train labels and media manifest order differ")
    if [str(row["id"]) for row in selector_labels] != [str(row["id"]) for row in selector_rows]:
        raise ValueError("selector labels and media manifest order differ")
    if _ordered_id_hash(train_labels) != receipt.get("train_id_order_sha256"):
        raise ValueError("train ID order differs from parent receipt")
    if _ordered_id_hash(selector_labels) != receipt.get("selector_id_order_sha256"):
        raise ValueError("selector ID order differs from parent receipt")
    rows = combine_train_selector_rows(train_rows, selector_rows)
    counts = receipt.get("counts", {})
    if len(train_rows) != counts.get("train") or len(selector_rows) != counts.get("selector"):
        raise ValueError("train/selector counts differ from parent receipt")

    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "train-selector-feature-manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    result = {
        "status": "FROZEN_SPATIAL_GRID_TRAIN_SELECTOR_INPUTS",
        "parent_input_receipt_sha256": sha256_file(receipt_path),
        "parent_config_sha256": receipt["config_sha256"],
        "train_count": len(train_rows),
        "selector_count": len(selector_rows),
        "feature_row_count": len(rows),
        "row_order": "train in frozen order followed by selector in frozen order",
        "train_id_order_sha256": _ordered_id_hash(train_labels),
        "selector_id_order_sha256": _ordered_id_hash(selector_labels),
        "feature_manifest_sha256": sha256_file(manifest),
        "unique_media_assets": len({str(row["media_sha256"]) for row in rows}),
        "train_selector_id_overlap": 0,
        "train_selector_media_overlap": 0,
        "reporting_partition_loaded": False,
        "sealed_audit_loaded": False,
    }
    receipt_out = output / "receipt.json"
    receipt_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["receipt_sha256"] = sha256_file(receipt_out)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(inputs=args.inputs, output=args.output)


if __name__ == "__main__":
    main()
