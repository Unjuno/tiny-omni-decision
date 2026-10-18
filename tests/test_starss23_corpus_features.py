import json

import pytest

from scripts.extract_frozen_videoprism_features import sha256_file
from scripts.extract_starss23_corpus_features import read_manifest


def create_manifest(tmp_path):
    report = {"status": "PASS", "splits": {}}
    for split in ("train", "calibration", "validation"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            json.dumps({"id": split, "recording": split, "room": split, "split": split}) + "\n"
        )
        report["splits"][split] = {"sha256": sha256_file(path), "windows": 1}
    (tmp_path / "result.json").write_text(json.dumps(report))
    return report


def test_complete_manifest_preserves_all_splits(tmp_path):
    create_manifest(tmp_path)
    assert [r["id"] for r in read_manifest(tmp_path)] == ["train", "calibration", "validation"]


def test_tampered_window_manifest_is_rejected(tmp_path):
    create_manifest(tmp_path)
    (tmp_path / "train.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_manifest(tmp_path)


def test_group_overlap_is_rejected_even_with_valid_file_hash(tmp_path):
    report = create_manifest(tmp_path)
    path = tmp_path / "validation.jsonl"
    row = json.loads(path.read_text())
    row["room"] = "train"
    path.write_text(json.dumps(row) + "\n")
    report["splits"]["validation"]["sha256"] = sha256_file(path)
    (tmp_path / "result.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="group overlap"):
        read_manifest(tmp_path)
