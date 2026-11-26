import numpy as np
import pytest

from tiny_omni_decision import mask_pair_association as module


def test_windows_cover_short_long_and_endpoint_without_self_time():
    assert module.association_windows(4, gap=2) == [
        (0, 1, 0),
        (1, 2, 0),
        (2, 3, 0),
        (0, 2, 1),
        (1, 3, 1),
        (0, 3, 2),
    ]
    windows = module.association_windows(32, gap=8)
    assert len(windows) == 56 and windows[-1] == (0, 31, 2)
    assert all(a < b for a, b, _ in windows)


def test_unknown_and_ambiguous_pairs_stay_in_eligibility_ledger():
    labels = np.array([1, 2, -1, 1, -2, 2, 1, 2], np.int64)
    offsets = np.array([0, 3, 6, 8], np.int64)
    pairs, targets, strata, ledger = module.labelled_mask_pairs(
        labels, offsets, [(0, 1, 0), (1, 2, 0), (0, 2, 1)]
    )
    assert pairs.tolist() == [
        [0, 3],
        [0, 5],
        [1, 3],
        [1, 5],
        [3, 6],
        [3, 7],
        [5, 6],
        [5, 7],
        [0, 6],
        [0, 7],
        [1, 6],
        [1, 7],
    ]
    assert targets.tolist() == [1, 0, 0, 1] * 3
    assert strata.tolist() == [0] * 8 + [1] * 4
    assert {
        k: ledger[k]
        for k in (
            "all_pairs",
            "labelled_pairs",
            "unknown_or_ambiguous_pairs",
            "unknown_pairs",
            "ambiguous_pairs",
        )
    } == dict(
        all_pairs=21,
        labelled_pairs=12,
        unknown_or_ambiguous_pairs=9,
        unknown_pairs=5,
        ambiguous_pairs=5,
    )


def test_each_temporal_stratum_retains_complete_exclusion_denominators():
    _, _, _, ledger = module.labelled_mask_pairs(
        np.array([1, 2, -1, 1, -2, 2, 1, 2], np.int64),
        np.array([0, 3, 6, 8], np.int64),
        [(0, 1, 0), (1, 2, 0), (0, 2, 1)],
    )
    assert ledger["unknown_and_ambiguous_pairs"] == 1
    assert ledger["strata"]["0"] == dict(
        all_pairs=15,
        labelled_pairs=8,
        unknown_or_ambiguous_pairs=7,
        unknown_pairs=3,
        ambiguous_pairs=5,
        unknown_and_ambiguous_pairs=1,
    )
    assert ledger["strata"]["1"] == dict(
        all_pairs=6,
        labelled_pairs=4,
        unknown_or_ambiguous_pairs=2,
        unknown_pairs=2,
        ambiguous_pairs=0,
        unknown_and_ambiguous_pairs=0,
    )
    assert ledger["strata"]["2"]["all_pairs"] == 0


def test_empty_frame_has_no_fabricated_pair_or_label():
    pairs, target, strata, ledger = module.labelled_mask_pairs(
        np.array([3], np.int64), np.array([0, 0, 1], np.int64), [(0, 1, 0)]
    )
    assert pairs.shape == (0, 2) and target.shape == strata.shape == (0,)
    assert ledger["all_pairs"] == ledger["labelled_pairs"] == 0


def test_scene_group_split_keeps_every_same_video_pair_together():
    train, heldout = module.grouped_pair_split(np.array([0, 1, 0, 2, 1], np.int64), 1)
    assert train.tolist() == [True, False, True, True, False]
    assert heldout.tolist() == [False, True, False, False, True]


def test_hand_checked_pair_features_are_symmetric_and_have_no_source_ids():
    appearance = np.array([[1, 3], [5, 7]], np.float32)
    boxes = np.array([[0, 0, 0.5, 0.25], [0.25, 0.25, 0.75, 0.5]], np.float64)
    times = np.array([0, 10], np.int64)
    before = [a.tobytes() for a in (appearance, boxes, times)]
    a = module.pair_features(appearance, boxes, times, np.array([[0, 1]], np.int64), 21)
    b = module.pair_features(appearance, boxes, times, np.array([[1, 0]], np.int64), 21)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, [[4, 4, 5, 21, 0.25, 0.25, 0, 0, 0.5]])
    assert a.tobytes() == b.tobytes()
    assert [a.tobytes() for a in (appearance, boxes, times)] == before


def test_standardizer_uses_training_pairs_only_and_handles_constants():
    x = np.array([[1, 8], [3, 8], [1000, 1000]], np.float32)
    selected = np.array([True, True, False])
    mean, scale = module.training_standardizer(x, selected)
    np.testing.assert_array_equal(mean, [2, 8])
    assert scale[0] == 1 and scale[1] == np.float32(1e-6)
    x[-1] = [-9999, -9999]
    other = module.training_standardizer(x, selected)
    assert all(a.tobytes() == b.tobytes() for a, b in zip((mean, scale), other, strict=True))


@pytest.mark.parametrize(
    "kind",
    [
        "bad_label",
        "float_label",
        "wrong_offsets",
        "self_window",
        "backward_window",
        "bad_window_index",
        "duplicate_window",
    ],
)
def test_corrupt_supervision_or_window_contract_is_rejected(kind):
    labels = np.array([1, 2], np.int64)
    offsets = np.array([0, 1, 2], np.int64)
    windows = [(0, 1, 0)]
    if kind == "bad_label":
        labels[0] = -3
    elif kind == "float_label":
        labels = labels.astype(np.float32)
    elif kind == "wrong_offsets":
        offsets[-1] = 1
    elif kind == "self_window":
        windows = [(0, 0, 0)]
    elif kind == "backward_window":
        windows = [(1, 0, 0)]
    elif kind == "bad_window_index":
        windows = [(0, 2, 0)]
    else:
        windows *= 2
    with pytest.raises(ValueError):
        module.labelled_mask_pairs(labels, offsets, windows)


@pytest.mark.parametrize("kind", ["nan", "zero_box", "float_time", "bad_pair", "bad_frames"])
def test_nonfinite_or_misaligned_pair_feature_input_fails_closed(kind):
    appearance = np.ones((2, 2), np.float32)
    boxes = np.array([[0, 0, 0.5, 0.5], [0, 0, 1, 1]], np.float64)
    times, pairs, frames = np.array([0, 1], np.int64), np.array([[0, 1]], np.int64), 2
    if kind == "nan":
        appearance[0, 0] = np.nan
    elif kind == "zero_box":
        boxes[0] = 0
    elif kind == "float_time":
        times = times.astype(np.float32)
    elif kind == "bad_pair":
        pairs[0, 1] = 2
    else:
        frames = 1
    with pytest.raises(ValueError):
        module.pair_features(appearance, boxes, times, pairs, frames)


def test_unrepresented_or_only_scene_cannot_be_held_out():
    for groups, heldout in [(np.array([0, 0], np.int64), 1), (np.array([0, 0], np.int64), 0)]:
        with pytest.raises(ValueError):
            module.grouped_pair_split(groups, heldout)


def test_scene_bootstrap_uses_group_means_and_keeps_constant_differences():
    rng = np.random.get_state()
    result = module.paired_scene_interval(np.full(8, 0.25, np.float64))
    assert result["mean"] == result["low"] == result["high"] == 0.25
    assert result["groups"] == 8 and result["replicates"] == 2000 and result["seed"] == 17
    varied = module.paired_scene_interval(np.array([-1.0, 1.0]))
    assert varied["mean"] == 0 and varied["low"] == -1 and varied["high"] == 1
    after = np.random.get_state()
    assert rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]
