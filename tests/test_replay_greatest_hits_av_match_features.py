import hashlib
import json

import pytest

from scripts.replay_greatest_hits_av_match_features import _ordered_feature_rows


def _write_feature_receipt(tmp_path, rows):
    manifest = tmp_path / "features.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "PASS_FROZEN_FEATURE_EXTRACTION_REPLAY",
                "features_manifest_sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def test_feature_replay_requires_exact_source_order_and_input_identity(tmp_path):
    rows = [
        {"source_group": "g1", "split": "train", "input_sha256": "a"},
        {"source_group": "g2", "split": "validation", "input_sha256": "b"},
    ]
    _write_feature_receipt(tmp_path, rows)

    result, actual = _ordered_feature_rows(tmp_path, rows)

    assert result["status"] == "PASS_FROZEN_FEATURE_EXTRACTION_REPLAY"
    assert actual == rows


def test_feature_replay_rejects_reordered_or_mismatched_inputs(tmp_path):
    rows = [
        {"source_group": "g1", "split": "train", "input_sha256": "a"},
        {"source_group": "g2", "split": "validation", "input_sha256": "b"},
    ]
    _write_feature_receipt(tmp_path, list(reversed(rows)))

    with pytest.raises(ValueError, match="order/source identity"):
        _ordered_feature_rows(tmp_path, rows)
