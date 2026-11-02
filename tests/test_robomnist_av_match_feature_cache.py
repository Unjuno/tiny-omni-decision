import hashlib
import json

import numpy as np
import pytest

from scripts.extract_robomnist_av_match_feature_cache import _read_receipt, group_order


def _manifest():
    return {
        "pair_units": {
            "train": [
                {"group_a": "a", "group_b": "b"},
                {"group_a": "a", "group_b": "c"},
            ],
            "validation": [{"group_a": "d", "group_b": "e"}],
        }
    }


def test_group_order_is_deterministic_and_grouped_by_split():
    groups, splits = group_order(_manifest())

    assert groups == ["a", "b", "c", "d", "e"]
    assert splits == {"train": ["a", "b", "c"], "validation": ["d", "e"]}


def test_group_order_rejects_cross_split_asset_overlap():
    manifest = _manifest()
    manifest["pair_units"]["validation"] = [{"group_a": "c", "group_b": "e"}]

    with pytest.raises(ValueError, match="cross train and validation"):
        group_order(manifest)


def test_feature_receipt_requires_matching_hash_id_shape_and_finite_values(tmp_path):
    group_id = "sample"
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(
        feature_path,
        group_id=np.asarray(group_id),
        video_features=np.zeros((8, 768), dtype=np.float32),
        audio_features=np.zeros((8, 384), dtype=np.float32),
    )
    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps({"status": "COMPLETE", "group_id": group_id, "feature_sha256": digest}),
        encoding="utf-8",
    )

    assert _read_receipt(receipt_path, feature_path, group_id)["group_id"] == group_id
    with pytest.raises(ValueError, match="invalid completion receipt"):
        _read_receipt(receipt_path, feature_path, "other")
