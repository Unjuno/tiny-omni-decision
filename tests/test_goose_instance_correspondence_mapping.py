from __future__ import annotations

import numpy as np

from scripts.probe_goose_instance_correspondence import instance_keys, load_instance_class_ids


def test_instance_keys_keep_only_instance_annotated_classes_and_include_id_zero():
    semantic = np.array([[12, 12, 5, 1]], dtype=np.uint8)
    instance = np.array([[0, 3, 99, 8]], dtype=np.uint8)

    assert instance_keys(semantic, instance, {1, 12}) == {(12, 0), (12, 3), (1, 8)}


def test_official_mapping_has_stable_instance_class_set(tmp_path):
    mapping = tmp_path / "goose_label_mapping.csv"
    mapping.write_text(
        "class_name,label_key,has_instance,hex\n"
        "undefined,0,0,#000000\n"
        "traffic_cone,1,1,#ffff00\n"
        "leaves,5,0,#008941\n"
        "car,12,1,#b79762\n",
        encoding="utf-8",
    )

    import hashlib

    expected = hashlib.sha256(mapping.read_bytes()).hexdigest()
    assert load_instance_class_ids(mapping, expected_sha256=expected) == {1, 12}


def test_mapping_hash_mismatch_fails_closed(tmp_path):
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("class_name,label_key,has_instance,hex\na,1,1,#ffffff\n")

    import pytest

    from scripts.probe_goose_instance_correspondence import load_instance_class_ids

    with pytest.raises(ValueError, match="SHA-256"):
        load_instance_class_ids(mapping)
