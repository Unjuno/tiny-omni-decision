from collections import Counter, defaultdict

import pytest

from tiny_omni_decision.synthetic_av_task import build_manifest, validate_manifest


def test_manifest_is_deterministic_and_query_polarities_are_complements():
    manifest = build_manifest(train_groups=4, validation_groups=2, seed=17)
    validate_manifest(manifest)
    assert manifest == build_manifest(train_groups=4, validation_groups=2, seed=17)
    rows = manifest["examples"]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], []).append(row)
    assert all(len(pair) == 2 for pair in grouped.values())
    assert all({row["target"] for row in pair} == {0, 1} for pair in grouped.values())


def test_manifest_makes_hard_negatives_with_same_multisets_and_balanced_marginals():
    manifest = build_manifest(train_groups=12, validation_groups=4, seed=17)
    rows = manifest["examples"]
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
        assert Counter(row["audio_symbols"]) == Counter(row["video_symbols"])
    for group_rows in by_group.values():
        assert len(group_rows) == 8
        for polarity in {row["query_polarity"] for row in group_rows}:
            selected = [row for row in group_rows if row["query_polarity"] == polarity]
            assert Counter(row["target"] for row in selected) == Counter({0: 2, 1: 2})


def test_manifest_rejects_sequence_identity_crossing_splits():
    manifest = build_manifest(train_groups=2, validation_groups=2, seed=17)
    validation_group = next(group for group in manifest["groups"] if group["split"] == "validation")
    train_group = next(group for group in manifest["groups"] if group["split"] == "train")
    validation_group["sequence_ids"]["A"] = train_group["sequence_ids"]["A"]
    with pytest.raises(ValueError, match="sequence crosses splits"):
        validate_manifest(manifest)


def test_manifest_rejects_missing_pair_examples():
    manifest = build_manifest(train_groups=2, validation_groups=2, seed=17)
    manifest["examples"].pop()
    with pytest.raises(ValueError, match="every group must have exactly eight examples"):
        validate_manifest(manifest)
