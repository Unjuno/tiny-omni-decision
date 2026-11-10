"""Freeze inputs and split receipts for the Clevr-4 VideoMAE Path A run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from tiny_omni_decision.dataset import sha256_file

EXPECTED_TRAIN_COUNT = 72_010
EXPECTED_SELECTOR_COUNT = 1_000
EXPECTED_REPORT_COUNT = 5_296


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _order_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(("\n".join(row["id"] for row in rows) + "\n").encode()).hexdigest()


def _copy_verified(source: Path, destination: Path) -> str:
    shutil.copyfile(source, destination)
    actual = sha256_file(destination)
    if actual != sha256_file(source):
        raise ValueError(f"copy hash mismatch for {source}")
    return actual


def freeze_inputs(
    *,
    train_labels: Path,
    train_media: Path,
    selector_labels: Path,
    selector_media: Path,
    report_labels: Path,
    report_media: Path,
    scene_audit: Path,
    validation_receipt: Path,
    config: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output {output}")
    train_y, train_x = _load_jsonl(train_labels), _load_jsonl(train_media)
    selector_y, selector_x = _load_jsonl(selector_labels), _load_jsonl(selector_media)
    report_y, report_x = _load_jsonl(report_labels), _load_jsonl(report_media)
    scene = json.loads(scene_audit.read_text(encoding="utf-8"))
    pool = json.loads(validation_receipt.read_text(encoding="utf-8"))
    effective = json.loads(config.read_text(encoding="utf-8"))
    if effective.get("experiment_id") != "path-a-image-videomae-clevr4-100k-seed17":
        raise ValueError("unexpected frozen experiment config")
    if (len(train_y), len(selector_y), len(report_y)) != (
        EXPECTED_TRAIN_COUNT,
        EXPECTED_SELECTOR_COUNT,
        EXPECTED_REPORT_COUNT,
    ):
        raise ValueError("frozen split counts differ from the pre-registered config")
    task = effective.get("task", {})
    if (
        task.get("training_images"),
        task.get("selector_images"),
        task.get("one_time_reporting_images"),
    ) != (
        EXPECTED_TRAIN_COUNT,
        EXPECTED_SELECTOR_COUNT,
        EXPECTED_REPORT_COUNT,
    ):
        raise ValueError("effective config split counts differ from input records")
    if pool.get("status") != "FROZEN_EXPOSURE_SCREENED_INTERNAL_VALIDATION_POOL":
        raise ValueError("validation pool receipt is missing or unexpected")
    if (
        pool.get("candidate_count") != 8_002
        or pool.get("quarantined_count") != 1_706
        or pool.get("selector_count") != EXPECTED_SELECTOR_COUNT
        or pool.get("report_count") != EXPECTED_REPORT_COUNT
    ):
        raise ValueError("validation pool receipt counts differ from the fixed protocol")
    if scene.get("status") != "COMPLETE_SCENE_CONFIGURATION_OVERLAP_AUDIT":
        raise ValueError("scene split audit is missing or unexpected")
    if (
        scene.get("train_count") != EXPECTED_TRAIN_COUNT
        or scene.get("validation_count") != 8_002
        or scene.get("matched_scene_records") != 80_012
        or "seed=17" not in scene.get("split_reproduction", "")
    ):
        raise ValueError("scene audit scope or deterministic split differs from this run")
    if (
        scene.get("exact_scene_config_cross_split_group_count") != 0
        or scene.get("rounded_3dp_scene_config_cross_split_group_count") != 0
    ):
        raise ValueError("scene audit reports cross-split duplicates")
    if any(
        len(y) != len(x) or [r["id"] for r in y] != [r["id"] for r in x]
        for y, x in ((train_y, train_x), (selector_y, selector_x), (report_y, report_x))
    ):
        raise ValueError("label/media manifest count or ID-order mismatch")
    for rows in (train_y, selector_y, report_y):
        if any(row.get("source_split") != "train" for row in rows):
            raise ValueError("input includes an official non-training source record")
    if sha256_file(selector_labels) != pool.get("selector_labels_sha256") or sha256_file(
        selector_media
    ) != pool.get("selector_media_manifest_sha256"):
        raise ValueError("selector files differ from the frozen exposure-screen receipt")
    if sha256_file(report_labels) != pool.get("report_labels_sha256") or sha256_file(
        report_media
    ) != pool.get("report_media_manifest_sha256"):
        raise ValueError("report files differ from the frozen exposure-screen receipt")
    if sha256_file(train_labels) != scene.get("train_labels_sha256") or sha256_file(
        train_media
    ) != scene.get("train_media_manifest_sha256"):
        raise ValueError("training files differ from the scene split audit")

    splits = {
        "train": (train_y, train_x),
        "selector": (selector_y, selector_x),
        "report": (report_y, report_x),
    }
    id_sets = {name: {row["id"] for row in labels} for name, (labels, _) in splits.items()}
    asset_sets = {name: {row["asset_id"] for row in labels} for name, (labels, _) in splits.items()}
    media_sets = {
        name: {row["media_sha256"] for row in media} for name, (_, media) in splits.items()
    }
    for name, (labels, media) in splits.items():
        if len(id_sets[name]) != len(labels) or len(media_sets[name]) != len(media):
            raise ValueError(f"duplicate ID or media identity inside {name} split")
        if len(asset_sets[name]) != len(labels):
            raise ValueError(f"duplicate underlying asset identity inside {name} split")
    for left, right in (("train", "selector"), ("train", "report"), ("selector", "report")):
        if (
            id_sets[left] & id_sets[right]
            or asset_sets[left] & asset_sets[right]
            or media_sets[left] & media_sets[right]
        ):
            raise ValueError(f"cross-split identity overlap: {left}/{right}")

    output.mkdir(parents=True, exist_ok=False)
    input_paths = {
        "train-labels.jsonl": train_labels,
        "train-media-manifest.jsonl": train_media,
        "selector-labels.jsonl": selector_labels,
        "selector-media-manifest.jsonl": selector_media,
        "report-labels.jsonl": report_labels,
        "report-media-manifest.jsonl": report_media,
    }
    file_hashes = {
        name: _copy_verified(source, output / name) for name, source in input_paths.items()
    }
    combined = [
        {**row, "subset": name} for name, (_, media_rows) in splits.items() for row in media_rows
    ]
    (output / "features-manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in combined), encoding="utf-8"
    )
    shutil.copyfile(config, output / "config.json")
    receipt = {
        "status": "FROZEN_VIDEOMAE_CLEVR4_PATH_A_INPUTS",
        "config_sha256": sha256_file(config),
        "file_hashes": file_hashes,
        "features_manifest_sha256": sha256_file(output / "features-manifest.jsonl"),
        "scene_audit_sha256": sha256_file(scene_audit),
        "validation_pool_receipt_sha256": sha256_file(validation_receipt),
        "counts": {name: len(labels) for name, (labels, _) in splits.items()},
        "unique_media_assets": len({row["media_sha256"] for row in combined}),
        "train_id_order_sha256": _order_hash(train_y),
        "selector_id_order_sha256": _order_hash(selector_y),
        "report_id_order_sha256": _order_hash(report_y),
        "cross_split_id_overlap": 0,
        "cross_split_asset_id_overlap": 0,
        "cross_split_media_sha256_overlap": 0,
        "official_validation_used": False,
        "sealed_audit_loaded": False,
        "source_files_modified": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-labels",
        "train-media",
        "selector-labels",
        "selector-media",
        "report-labels",
        "report-media",
        "scene-audit",
        "validation-receipt",
        "config",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_inputs(
                train_labels=args.train_labels,
                train_media=args.train_media,
                selector_labels=args.selector_labels,
                selector_media=args.selector_media,
                report_labels=args.report_labels,
                report_media=args.report_media,
                scene_audit=args.scene_audit,
                validation_receipt=args.validation_receipt,
                config=args.config,
                output=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
