import hashlib
import json

import pytest

from scripts import extract_greatest_hits_av_match_features as feature_runner


def _inputs(tmp_path):
    group_manifest = tmp_path / "groups.jsonl"
    group_manifest.write_text(
        json.dumps({"source_group": "group-1", "split": "train"}) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(group_manifest.read_bytes()).hexdigest()
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS_FULL_DECODE_AND_NORMALIZED_CONTENT_SPLIT_GATES",
                "completed_groups": 1,
                "artifact_hashes": {"groups.jsonl": digest},
            }
        ),
        encoding="utf-8",
    )
    return group_manifest


def test_feature_runner_requires_complete_input_gate_and_pinned_groups_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(
        feature_runner, "EXPECTED_GROUPS", {"train": 1, "calibration": 0, "validation": 0}
    )
    group_manifest = _inputs(tmp_path)

    result, rows = feature_runner.read_groups(tmp_path)

    assert result["completed_groups"] == 1
    assert rows == [{"source_group": "group-1", "split": "train"}]
    group_manifest.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        feature_runner.read_groups(tmp_path)
