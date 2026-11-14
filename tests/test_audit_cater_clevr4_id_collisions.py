import json

import pytest

from scripts.audit_cater_clevr4_id_collisions import (
    audit_split,
    read_cater_list,
    read_clevr4_indices,
)


def test_cater_id_collision_is_excluded_and_class_counts_are_preserved(tmp_path):
    path = tmp_path / "train.txt"
    path.write_text("CATER_new_000001.avi 0\nCATER_new_000002.avi 1\n", encoding="utf-8")

    rows = read_cater_list(path)
    result = audit_split(rows, {1})

    assert result["rows"] == 2
    assert result["index_collisions"] == 1
    assert result["excluded_video_ids"] == [1]
    assert result["kept_after_conservative_exclusion"] == 1
    assert result["kept_per_class"]["1"] == 1
    assert result["missing_classes"] == [label for label in range(64) if label != 1]


def test_cater_list_rejects_duplicate_ids_and_out_of_range_labels(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("CATER_new_000001.avi 0\nCATER_new_000001.avi 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate CATER video id"):
        read_cater_list(path)

    path.write_text("CATER_new_000002.avi 64\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label outside"):
        read_cater_list(path)


def test_clevr4_manifest_requires_unique_asset_indices(tmp_path):
    path = tmp_path / "manifest.jsonl"
    rows = [
        {"asset_id": "CLEVR_new_000001"},
        {"asset_id": "CLEVR_new_000001"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate Clevr-4 asset index"):
        read_clevr4_indices(path)
