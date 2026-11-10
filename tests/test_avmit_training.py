from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")

from tiny_omni_decision.avmit_training import (
    FeatureStore,
    PairGroup,
    _metrics,
    load_pair_groups,
    paired_group_bootstrap,
    verify_unimodal_balance,
)


def _group(index: int) -> PairGroup:
    assets = [f"clip{index}-{value}" for value in range(4)]
    entries = (
        (0, 1, "same_action"),
        (1, 0, "same_action"),
        (2, 3, "same_action"),
        (3, 2, "same_action"),
        (0, 2, "different_action"),
        (2, 1, "different_action"),
        (1, 3, "different_action"),
        (3, 0, "different_action"),
    )
    rows = tuple(
        {
            "pair_group_sha256": str(index),
            "audio_asset_id_sha256": assets[a],
            "video_asset_id_sha256": assets[v],
            "target": target,
        }
        for a, v, target in entries
    )
    return PairGroup(str(index), rows)


def test_load_pair_groups_and_verify_unimodal_balance(tmp_path):
    group = _group(1)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in group.rows), encoding="utf-8"
    )

    groups = load_pair_groups(manifest)

    assert len(groups) == 1
    assert verify_unimodal_balance(groups) == {
        "groups": 1,
        "rows": 8,
        "unimodal_accuracy": 0.5,
    }


def test_unimodal_verifier_rejects_missing_cross_class_edges():
    group = _group(1)
    rows = list(group.rows)
    rows[5] = {**rows[5], "video_asset_id_sha256": rows[4]["video_asset_id_sha256"]}
    malformed = PairGroup(group.group_id, tuple(rows))

    try:
        verify_unimodal_balance([malformed])
    except ValueError as error:
        assert "all four cross-label" in str(error)
    else:
        raise AssertionError("incomplete cross-label edge set was accepted")


def test_feature_store_reads_exact_indexed_temporal_slices(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    audio = np.arange(12, dtype="<f4")
    video = np.arange(18, dtype="<f4")
    audio.tofile(split / "audio.f32")
    video.tofile(split / "video.f32")
    row = {
        "asset_id_sha256": "asset",
        "audio_offset_bytes": 4 * 4,
        "audio_steps": 2,
        "audio_width": 2,
        "video_offset_bytes": 4 * 4,
        "video_steps": 2,
        "video_width": 3,
    }
    (split / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    store = FeatureStore(split)

    np.testing.assert_array_equal(store.sequence("asset", "audio"), [[4, 5], [6, 7]])
    np.testing.assert_array_equal(store.sequence("asset", "video"), [[4, 5, 6], [7, 8, 9]])


def test_binary_metrics_include_confusion_and_group_bootstrap_is_seeded():
    logits = np.asarray([[3.0, 0.0], [0.0, 3.0], [0.0, 3.0], [3.0, 0.0]])
    targets = np.asarray([0, 0, 1, 1])
    metrics, probabilities = _metrics(logits, targets)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert probabilities.shape == (4, 2)

    groups = [_group(index) for index in range(3)]
    all_targets = np.tile(np.asarray([0, 0, 0, 0, 1, 1, 1, 1]), 3)
    mean_logits = np.zeros((24, 2), dtype=np.float64)
    gru_logits = mean_logits.copy()
    gru_logits[:, 0] = 0.25
    first = paired_group_bootstrap(mean_logits, gru_logits, all_targets, groups, replicates=50)
    second = paired_group_bootstrap(mean_logits, gru_logits, all_targets, groups, replicates=50)
    assert first == second
    assert first["replicates"] == 50
    assert first["unit"] == "pair_group (all eight rows sampled together)"


def test_group_loader_rejects_incomplete_pair_groups(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(json.dumps(_group(1).rows[0]) + "\n", encoding="utf-8")
    try:
        load_pair_groups(manifest)
    except ValueError as error:
        assert "exactly eight" in str(error)
    else:
        raise AssertionError("incomplete pair group was accepted")


def test_group_loader_rejects_reused_source_across_groups(tmp_path):
    first = _group(1)
    second = _group(2)
    second_rows = [
        {
            **row,
            "audio_asset_id_sha256": row["audio_asset_id_sha256"].replace("clip2-0", "clip1-0"),
            "video_asset_id_sha256": row["video_asset_id_sha256"].replace("clip2-0", "clip1-0"),
        }
        for row in second.rows
    ]
    manifest = tmp_path / "overlap.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in (*first.rows, *second_rows)),
        encoding="utf-8",
    )

    try:
        load_pair_groups(manifest)
    except ValueError as error:
        assert "more than one pair group" in str(error)
    else:
        raise AssertionError("reused source clip was accepted")
