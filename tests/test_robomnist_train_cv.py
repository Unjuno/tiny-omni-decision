from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from tiny_omni_decision.robomnist_train_cv import grouped_folds, pooled_normalize


def _pair(index, condition="c"):
    a, b = f"a{index}", f"b{index}"
    return [
        {
            "pair_unit_id": f"pair{index}",
            "condition": condition,
            "audio_group": audio,
            "video_group": video,
            "target": int(audio == video),
        }
        for audio, video in ((a, a), (a, b), (b, a), (b, b))
    ]


def test_folds_keep_groups_together_and_are_stratified_balanced_and_order_invariant():
    rows = [row for i in range(23) for row in _pair(i, condition=f"c{i % 4}")]
    assignments = grouped_folds(rows)
    assert assignments == grouped_folds(list(reversed(rows)))
    assert assignments != grouped_folds(rows, seed=18)
    counts = Counter(assignments.values())
    assert set(counts) == set(range(5))
    assert max(counts.values()) - min(counts.values()) <= 1
    for condition in {row["condition"] for row in rows}:
        ids = {row["pair_unit_id"] for row in rows if row["condition"] == condition}
        counts = [sum(assignments[pair] == fold for pair in ids) for fold in range(5)]
        assert max(counts) - min(counts) <= 1
    source_folds = {}
    for row in rows:
        for key in ("audio_group", "video_group"):
            source_folds.setdefault(row[key], set()).add(assignments[row["pair_unit_id"]])
    assert all(len(folds) == 1 for folds in source_folds.values())


def test_folds_reject_source_reuse_even_with_distinct_sample_pairs():
    rows = [row for i in range(5) for row in _pair(i)]
    # The first changed row has a new AV pair, but an existing source identity.
    rows[4]["audio_group"] = "a0"
    rows[4]["target"] = 0
    with pytest.raises(ValueError, match="multiple pair units"):
        grouped_folds(rows)


def test_folds_reject_duplicate_samples_incomplete_pairs_and_wrong_targets():
    rows = [row for i in range(5) for row in _pair(i)]
    with pytest.raises(ValueError, match="duplicate TRAIN sample"):
        grouped_folds(rows + [deepcopy(rows[0])])
    with pytest.raises(ValueError, match="complete balanced 2x2"):
        grouped_folds(rows[:-1])
    bad_target = deepcopy(rows)
    bad_target[0]["target"] = 0
    with pytest.raises(ValueError, match="target disagrees"):
        grouped_folds(bad_target)
    explicit_ids = deepcopy(rows)
    for row in explicit_ids:
        row["sample_id"] = "same-id"
    with pytest.raises(ValueError, match="sample_id"):
        grouped_folds(explicit_ids)


def test_held_fold_outlier_cannot_change_fit_normalization():
    rng = np.random.default_rng(17)
    features = {
        group: (rng.normal(size=(8, 3)).astype("float32"),
                rng.normal(size=(8, 4)).astype("float32"))
        for group in ("train_a", "train_b", "held")
    }
    fit_groups = {"train_a", "train_b"}
    original, original_norm = pooled_normalize(features, fit_groups)
    changed = deepcopy(features)
    changed["held"] = tuple(stream * 1000 + 1000 for stream in changed["held"])
    transformed, changed_norm = pooled_normalize(changed, fit_groups)
    for key in original_norm:
        np.testing.assert_array_equal(original_norm[key], changed_norm[key])
    for group in fit_groups:
        for expected, actual in zip(original[group], transformed[group], strict=True):
            np.testing.assert_array_equal(expected, actual)
    assert not np.array_equal(original["held"][0], transformed["held"][0])
    # Independent explicit baseline formula, fitted on unique sources only.
    pooled = np.stack([
        np.concatenate((features[group][0].mean(0), features[group][0].std(0)))
        for group in sorted(fit_groups)
    ])
    np.testing.assert_array_equal(original_norm["audio_mean"], pooled.mean(0))
    np.testing.assert_array_equal(original_norm["audio_scale"], pooled.std(0))


def test_zero_variance_normalization_is_finite_and_missing_fit_groups_fail():
    features = {"a": (np.ones((8, 2)), np.ones((8, 3)))}
    normalized, normalizers = pooled_normalize(features, {"a"})
    assert all(np.all(stream == 0) for stream in normalized["a"])
    assert all(np.all(normalizers[key] == 1) for key in ("audio_scale", "video_scale"))
    with pytest.raises(ValueError, match="subset"):
        pooled_normalize(features, {"missing"})
