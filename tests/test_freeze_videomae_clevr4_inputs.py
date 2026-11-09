from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import freeze_videomae_clevr4_inputs as freeze_module
from scripts.audit_clevr4_content_overlap import sha256_file


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    train_labels = [{"id": "train-0", "asset_id": "asset-train", "source_split": "train"}]
    selector_labels = [{"id": "selector-0", "asset_id": "asset-selector", "source_split": "train"}]
    report_labels = [{"id": "report-0", "asset_id": "asset-report", "source_split": "train"}]
    train_media = [{"id": "train-0", "media_sha256": "media-train"}]
    selector_media = [{"id": "selector-0", "media_sha256": "media-selector"}]
    report_media = [{"id": "report-0", "media_sha256": "media-report"}]
    contents = {
        "train-labels.jsonl": train_labels,
        "selector-labels.jsonl": selector_labels,
        "report-labels.jsonl": report_labels,
        "train-media.jsonl": train_media,
        "selector-media.jsonl": selector_media,
        "report-media.jsonl": report_media,
    }
    paths: dict[str, Path] = {}
    for name, rows in contents.items():
        path = tmp_path / name
        _write_jsonl(path, rows)
        paths[name] = path

    paths["scene"] = tmp_path / "scene.json"
    paths["scene"].write_text(
        json.dumps(
            {
                "status": "COMPLETE_SCENE_CONFIGURATION_OVERLAP_AUDIT",
                "train_count": 1,
                "validation_count": 8_002,
                "matched_scene_records": 80_012,
                "split_reproduction": "PASS seed=17",
                "exact_scene_config_cross_split_group_count": 0,
                "rounded_3dp_scene_config_cross_split_group_count": 0,
                "train_labels_sha256": sha256_file(paths["train-labels.jsonl"]),
                "train_media_manifest_sha256": sha256_file(paths["train-media.jsonl"]),
            }
        ),
        encoding="utf-8",
    )
    paths["pool"] = tmp_path / "pool.json"
    paths["pool"].write_text(
        json.dumps(
            {
                "status": "FROZEN_EXPOSURE_SCREENED_INTERNAL_VALIDATION_POOL",
                "candidate_count": 8_002,
                "quarantined_count": 1_706,
                "selector_count": 1,
                "report_count": 1,
                "selector_labels_sha256": sha256_file(paths["selector-labels.jsonl"]),
                "selector_media_manifest_sha256": sha256_file(paths["selector-media.jsonl"]),
                "report_labels_sha256": sha256_file(paths["report-labels.jsonl"]),
                "report_media_manifest_sha256": sha256_file(paths["report-media.jsonl"]),
            }
        ),
        encoding="utf-8",
    )
    paths["config"] = tmp_path / "config.json"
    paths["config"].write_text(
        json.dumps(
            {
                "experiment_id": "path-a-image-videomae-clevr4-100k-seed17",
                "task": {
                    "training_images": 1,
                    "selector_images": 1,
                    "one_time_reporting_images": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    paths["output"] = tmp_path / "frozen"
    return paths


def _freeze(paths: dict[str, Path]) -> dict[str, object]:
    return freeze_module.freeze_inputs(
        train_labels=paths["train-labels.jsonl"],
        train_media=paths["train-media.jsonl"],
        selector_labels=paths["selector-labels.jsonl"],
        selector_media=paths["selector-media.jsonl"],
        report_labels=paths["report-labels.jsonl"],
        report_media=paths["report-media.jsonl"],
        scene_audit=paths["scene"],
        validation_receipt=paths["pool"],
        config=paths["config"],
        output=paths["output"],
    )


def test_freeze_inputs_hashes_and_checks_disjoint_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(freeze_module, "EXPECTED_TRAIN_COUNT", 1)
    monkeypatch.setattr(freeze_module, "EXPECTED_SELECTOR_COUNT", 1)
    monkeypatch.setattr(freeze_module, "EXPECTED_REPORT_COUNT", 1)
    paths = _fixture(tmp_path)

    receipt = _freeze(paths)

    assert receipt["cross_split_id_overlap"] == 0
    assert receipt["cross_split_asset_id_overlap"] == 0
    assert receipt["cross_split_media_sha256_overlap"] == 0
    assert receipt["unique_media_assets"] == 3
    assert (paths["output"] / "features-manifest.jsonl").is_file()


def test_freeze_inputs_rejects_reporting_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(freeze_module, "EXPECTED_TRAIN_COUNT", 1)
    monkeypatch.setattr(freeze_module, "EXPECTED_SELECTOR_COUNT", 1)
    monkeypatch.setattr(freeze_module, "EXPECTED_REPORT_COUNT", 1)
    paths = _fixture(tmp_path)
    overlapping = [{"id": "train-0", "asset_id": "asset-report", "source_split": "train"}]
    _write_jsonl(paths["report-labels.jsonl"], overlapping)
    _write_jsonl(paths["report-media.jsonl"], [{"id": "train-0", "media_sha256": "media-report"}])
    paths["pool"].write_text(
        json.dumps(
            {
                "status": "FROZEN_EXPOSURE_SCREENED_INTERNAL_VALIDATION_POOL",
                "candidate_count": 8_002,
                "quarantined_count": 1_706,
                "selector_count": 1,
                "report_count": 1,
                "selector_labels_sha256": sha256_file(paths["selector-labels.jsonl"]),
                "selector_media_manifest_sha256": sha256_file(paths["selector-media.jsonl"]),
                "report_labels_sha256": sha256_file(paths["report-labels.jsonl"]),
                "report_media_manifest_sha256": sha256_file(paths["report-media.jsonl"]),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cross-split identity overlap"):
        _freeze(paths)
