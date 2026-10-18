import copy

import numpy as np
import pytest

from tiny_omni_decision import mask_feature_corpus as module


def plan():
    return dict(
        video_id="video_x",
        split="train",
        frames=9,
        cut_frame=9,
        media_path="source.mp4",
        media_sha256="fixed",
        frame_indices=[0, 4, 8],
    )


def frame(ids, means):
    n = len(ids)
    return dict(
        mean=np.asarray(means, np.float32).reshape(n, 2),
        std=np.zeros((n, 2), np.float32),
        foreground_counts=np.ones(n, np.int64),
        proposal_ids=np.array(ids, np.int64),
        boxes=np.zeros((n, 4), np.float64),
        scores=np.arange(n, dtype=np.float32),
    )


def test_different_asset_cut_split_or_observation_time_is_rejected():
    p = plan()
    for key, value in [
        ("video_id", "other"),
        ("split", "validation"),
        ("frames", 10),
        ("cut_frame", -1),
        ("media_sha256", "changed"),
        ("media_path", "other"),
    ]:
        fine = copy.deepcopy(p)
        fine[key] = value
        with pytest.raises(ValueError):
            module.require_aligned_feature_times(p, fine, np.array([0, 4, 8], np.int64))
    with pytest.raises(ValueError):
        module.require_aligned_feature_times(p, p, np.array([0, 3, 8], np.int64))


def test_full_source_cut_minus_one_and_ordered_times_are_retained():
    p = plan()
    p["cut_frame"] = -1
    before = copy.deepcopy(p)
    module.require_aligned_feature_times(p, p, np.array([0, 4, 8], np.int64))
    assert p == before


@pytest.mark.parametrize(
    "times",
    [
        np.array([0.0, 4.0, 8.0]),
        np.array([0, 4, 4]),
        np.array([0, 8, 4]),
        np.array([0, 4]),
        np.array([0, 4, 8], np.int32),
    ],
)
def test_invalid_time_dtype_count_or_order_fails_closed(times):
    with pytest.raises(ValueError):
        module.require_aligned_feature_times(plan(), plan(), times)


def test_all_proposals_keep_native_order_and_empty_frame_offset():
    frames = [frame([7, 2], [[1, 2], [3, 4]]), frame([], []), frame([5], [[9, 10]])]
    before = [[a.tobytes() for a in f.values()] for f in frames]
    rng = np.random.get_state()
    a = module.assemble_mask_statistics(np.array([0, 4, 8], np.int64), frames)
    assert np.array_equal(a["frame_offsets"], [0, 2, 2, 3])
    assert np.array_equal(a["frame_indices"], [0, 0, 8])
    assert np.array_equal(a["proposal_ids"], [7, 2, 5])
    assert np.array_equal(a["mean"], [[1, 2], [3, 4], [9, 10]])
    assert np.array_equal(a["observation_times"], [0, 4, 8])
    assert len(a) == 9 and a["frame_indices"].dtype == np.int64
    assert [[v.tobytes() for v in f.values()] for f in frames] == before
    after = np.random.get_state()
    assert rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]


def test_same_native_proposal_id_at_different_times_is_not_deduplicated():
    a = module.assemble_mask_statistics(
        np.array([1, 3], np.int64), [frame([2], [[1, 2]]), frame([2], [[3, 4]])]
    )
    assert np.array_equal(a["proposal_ids"], [2, 2])
    assert np.array_equal(a["frame_offsets"], [0, 1, 2])


def test_empty_mask_zero_statistics_and_zero_count_are_preserved():
    f = frame([2], [[0, 0]])
    f["foreground_counts"][0] = 0
    a = module.assemble_mask_statistics(np.array([1], np.int64), [f])
    assert a["foreground_counts"].tolist() == [0] and a["proposal_ids"].tolist() == [2]


def test_native_float64_boxes_keep_exact_bytes_without_a_cast():
    f = frame([2], [[1, 2]])
    f["boxes"][0] = [0.12345678912345, 0.2, 0.7, 0.8]
    a = module.assemble_mask_statistics(np.array([1], np.int64), [f])
    assert a["boxes"].dtype == np.float64 and a["boxes"].tobytes() == f["boxes"].tobytes()


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_id",
        "negative_id",
        "half_mean",
        "nan_score",
        "negative_std",
        "wrong_box_count",
        "wrong_channels",
        "negative_mass",
        "nonzero_empty",
        "missing_key",
        "extra_key",
    ],
)
def test_corrupt_frame_inventory_is_rejected(kind):
    f = frame([7, 2], [[1, 2], [3, 4]])
    if kind == "duplicate_id":
        f["proposal_ids"][1] = 7
    elif kind == "negative_id":
        f["proposal_ids"][0] = -1
    elif kind == "half_mean":
        f["mean"] = f["mean"].astype(np.float16)
    elif kind == "nan_score":
        f["scores"][0] = np.nan
    elif kind == "negative_std":
        f["std"][0, 0] = -1
    elif kind == "wrong_box_count":
        f["boxes"] = f["boxes"][:1]
    elif kind == "wrong_channels":
        f["std"] = f["std"][:, :1]
    elif kind == "negative_mass":
        f["foreground_counts"][0] = -1
    elif kind == "nonzero_empty":
        f["foreground_counts"][0] = 0
    elif kind == "missing_key":
        del f["scores"]
    else:
        f["target"] = np.ones(2)
    with pytest.raises(ValueError):
        module.assemble_mask_statistics(np.array([1], np.int64), [f])


def test_missing_frames_and_inconsistent_channel_counts_are_rejected():
    f = frame([2], [[1, 2]])
    with pytest.raises(ValueError):
        module.assemble_mask_statistics(np.array([1, 3], np.int64), [f])
    other = frame([3], [[3, 4]])
    other["mean"] = other["mean"][:, :1]
    other["std"] = other["std"][:, :1]
    with pytest.raises(ValueError):
        module.assemble_mask_statistics(np.array([1, 3], np.int64), [f, other])
