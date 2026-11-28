from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_physionpp_split_integrity import audit_train_validation_splits


def _row(
    *, split: str, sample_id: str, source_id: str, asset_group: str, marker: str
) -> dict[str, object]:
    return {
        "id": sample_id,
        "modality": "video",
        "state": "A simulated physical-scene video is supplied.",
        "question": "Did the red object contact the yellow-marked target at any point in the clip?",
        "options": ["No", "Yes"],
        "target": "No",
        "media": [
            {
                "kind": "video",
                "path": f"raw/{marker}.mp4",
                "frame_refs": [],
                "sha256": marker * 64,
                "license": "MIT",
            }
        ],
        "source": "physionpp/readout",
        "source_revision": "0c53b983ca138962ba636c0936e735490ad4c326",
        "source_record_id": source_id,
        "split": split,
        "source_asset_group_id": asset_group,
        "provenance": {"license": "MIT", "trust_status": "trusted"},
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_split_audit_accepts_disjoint_records_scenes_and_media(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write(
        train_path,
        [
            _row(
                split="train",
                sample_id="train-1",
                source_id="record-1",
                asset_group="scene-1",
                marker="a",
            )
        ],
    )
    _write(
        validation_path,
        [
            _row(
                split="validation",
                sample_id="validation-1",
                source_id="record-2",
                asset_group="scene-2",
                marker="b",
            )
        ],
    )

    result = audit_train_validation_splits(train_path, validation_path)

    assert result["status"] == "disjoint"
    assert result["train_records"] == 1
    assert result["validation_records"] == 1
    assert result["overlaps"]["source_asset_group_id"]["shared"] == 0
    assert result["overlaps"]["media_sha256"]["shared"] == 0
    assert result["overlaps"]["normalized_content_and_source_id"]["status"] == "disjoint"
    assert result["media_decoded"] is False
    assert result["model_loaded"] is False


@pytest.mark.parametrize("shared_key", ["source_asset_group_id", "source_record_id", "media"])
def test_split_audit_rejects_identity_overlap(tmp_path: Path, shared_key: str) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train = _row(
        split="train", sample_id="train-1", source_id="record-1", asset_group="scene-1", marker="a"
    )
    validation = _row(
        split="validation",
        sample_id="validation-1",
        source_id="record-2",
        asset_group="scene-2",
        marker="b",
    )
    if shared_key == "source_asset_group_id":
        validation["source_asset_group_id"] = train["source_asset_group_id"]
    elif shared_key == "source_record_id":
        validation["source_record_id"] = train["source_record_id"]
    else:
        validation["media"] = train["media"]
    _write(train_path, [train])
    _write(validation_path, [validation])

    with pytest.raises(ValueError):
        audit_train_validation_splits(train_path, validation_path)


def test_split_audit_rejects_wrong_split_label(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write(
        train_path,
        [
            _row(
                split="validation",
                sample_id="x",
                source_id="record-1",
                asset_group="scene-1",
                marker="a",
            )
        ],
    )
    _write(
        validation_path,
        [
            _row(
                split="validation",
                sample_id="validation-1",
                source_id="record-2",
                asset_group="scene-2",
                marker="b",
            )
        ],
    )

    with pytest.raises(ValueError, match="outside the train split"):
        audit_train_validation_splits(train_path, validation_path)


def test_split_audit_rejects_duplicate_source_record_within_split(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    first = _row(
        split="train", sample_id="train-1", source_id="record-1", asset_group="scene-1", marker="a"
    )
    second = _row(
        split="train", sample_id="train-2", source_id="record-1", asset_group="scene-2", marker="b"
    )
    _write(train_path, [first, second])
    _write(
        validation_path,
        [
            _row(
                split="validation",
                sample_id="validation-1",
                source_id="record-3",
                asset_group="scene-3",
                marker="c",
            )
        ],
    )

    with pytest.raises(ValueError, match="source record IDs must be unique"):
        audit_train_validation_splits(train_path, validation_path)


@pytest.mark.parametrize("media_key", ["sha256", "path"])
def test_split_audit_rejects_duplicate_media_within_split(
    tmp_path: Path, media_key: str
) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    first = _row(
        split="train", sample_id="train-1", source_id="record-1", asset_group="scene-1", marker="a"
    )
    second = _row(
        split="train", sample_id="train-2", source_id="record-2", asset_group="scene-2", marker="b"
    )
    second["media"][0][media_key] = first["media"][0][media_key]
    _write(train_path, [first, second])
    _write(
        validation_path,
        [
            _row(
                split="validation",
                sample_id="validation-1",
                source_id="record-3",
                asset_group="scene-3",
                marker="c",
            )
        ],
    )

    with pytest.raises(ValueError, match=f"video {media_key} values must be unique"):
        audit_train_validation_splits(train_path, validation_path)
