import hashlib

import numpy as np

from tiny_omni_decision.subsevir_features import (
    cls_plus_patch_mean,
    frame_to_rgb,
    required_frame_indices,
)
from tiny_omni_decision.subsevir_task import (
    build_forecast_windows,
    build_fresh_training_manifest,
    fresh_training_event_split,
)


def _rows(event_id: str, start: int, labels: list[int], *, gap_at: int | None = None):
    timestamps = [start + 5 * index for index in range(len(labels))]
    if gap_at is not None:
        timestamps[gap_at:] = [value + 2 for value in timestamps[gap_at:]]
    return ([event_id] * len(labels), timestamps, labels)


def test_forecast_windows_use_eight_past_frames_and_next_label():
    ids, timestamps, labels = _rows("storm-a", 0, [0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    windows = build_forecast_windows(ids, timestamps, labels, seed=17)

    assert len(windows) == 2
    assert windows[0].input_indices == tuple(range(8))
    assert windows[0].target_index == 8
    assert windows[0].target == 1
    assert windows[1].input_indices == tuple(range(1, 9))
    assert windows[1].target_index == 9
    assert windows[1].target == 0
    assert windows[0].split == windows[1].split


def test_irregular_cadence_excludes_only_windows_touching_the_gap():
    ids, timestamps, labels = _rows("storm-b", 100, [index % 2 for index in range(12)], gap_at=9)
    windows = build_forecast_windows(ids, timestamps, labels)

    # The gap first enters the 9-point sequence for target 9 and excludes
    # each later target window; target 8 remains regular.
    assert [window.target_index for window in windows] == [8]


def test_event_split_is_deterministic_and_never_splits_one_event():
    rows = [_rows(f"storm-{index}", index * 100, [0, 1] * 5) for index in range(30)]
    ids = sum((item[0] for item in rows), [])
    timestamps = sum((item[1] for item in rows), [])
    labels = sum((item[2] for item in rows), [])

    first = build_forecast_windows(ids, timestamps, labels, seed=17)
    second = build_forecast_windows(ids, timestamps, labels, seed=17)
    assert first == second
    by_event = {}
    for window in first:
        by_event.setdefault(window.event_id, set()).add(window.split)
    assert all(len(splits) == 1 for splits in by_event.values())
    assert {window.split for window in first} == {"train", "validation"}


def test_fresh_training_split_is_deterministic_and_namespaced():
    payload = b"subsevir-gated-fresh-train-split-v1\0" + b"17\0storm-a"
    digest = hashlib.sha256(payload).digest()
    expected = "validation" if int.from_bytes(digest[:8], "big") / 2**64 < 0.2 else "train"
    assert fresh_training_event_split("storm-a", seed=17) == expected
    assert fresh_training_event_split("storm-a", seed=17) == fresh_training_event_split(
        "storm-a", seed=17
    )


def test_fresh_manifest_uses_only_source_train_events_and_groups_windows():
    train_ids = [f"train-storm-{index:03d}" for index in range(100)]
    old_validation_ids = [f"old-validation-storm-{index:03d}" for index in range(12)]
    windows = []
    for event_id in train_ids + old_validation_ids:
        source_split = "train" if event_id in train_ids else "validation"
        for offset in (8, 9):
            target_index = offset
            windows.append({
                "event_id": event_id,
                "split": source_split,
                "input_indices": list(range(offset - 8, offset)),
                "target_index": target_index,
                "target": offset % 2,
                "sample_id": f"{event_id}:target-{target_index}",
            })
    parent = {
        "split_group_ids": {
            "train": train_ids,
            "validation": old_validation_ids,
        },
        "windows": windows,
    }

    first = build_fresh_training_manifest(parent, parent_manifest_sha256="a" * 64)
    second = build_fresh_training_manifest(parent, parent_manifest_sha256="a" * 64)

    assert first == second
    assert first["source_train_group_count"] == len(train_ids)
    assert first["source_validation_group_count_excluded"] == len(old_validation_ids)
    assert first["split_window_counts"]["train"] + first["split_window_counts"]["validation"] == 200
    assert set(first["split_group_ids"]["train"]).isdisjoint(
        first["split_group_ids"]["validation"]
    )
    assert set(first["split_group_ids"]["train"]) | set(
        first["split_group_ids"]["validation"]
    ) == set(train_ids)
    new_validation = set(first["split_group_ids"]["validation"])
    assert new_validation.isdisjoint(old_validation_ids)
    by_event = {}
    for window in first["windows"]:
        by_event.setdefault(window["event_id"], set()).add(window["split"])
        assert window["event_id"] in train_ids
    assert all(len(splits) == 1 for splits in by_event.values())
    assert len(first["windows"]) == len(windows) - 2 * len(old_validation_ids)


def test_rejects_noncontiguous_rows_for_same_event():
    ids = ["storm-a"] * 9 + ["storm-b"] * 9 + ["storm-a"]
    timestamps = list(range(len(ids)))
    labels = [0] * len(ids)
    try:
        build_forecast_windows(ids, timestamps, labels)
    except ValueError as exc:
        assert "not contiguous" in str(exc)
    else:
        raise AssertionError("expected a noncontiguous-event error")


def test_frame_mapping_uses_fixed_train_ranges_and_stream_channels():
    frame = np.zeros((2, 2, 4), dtype=np.float32)
    frame[..., 0] = 0.5
    frame[..., 1] = 0.25
    frame[..., 2] = 1.0
    frame[..., 3] = 0.75
    ranges = [[0.0, 1.0]] * 4

    satellite = frame_to_rgb(frame, ranges, "satellite")
    radar = frame_to_rgb(frame, ranges, "radar")
    assert satellite.shape == (2, 2, 3)
    assert satellite[0, 0].tolist() == [128, 64, 191]
    assert radar[0, 0].tolist() == [255, 255, 255]


def test_required_frame_indices_are_unique_and_split_scoped():
    windows = [
        {"split": "train", "event_id": "a", "input_indices": list(range(8))},
        {"split": "train", "event_id": "a", "input_indices": list(range(1, 9))},
        {"split": "validation", "event_id": "b", "input_indices": list(range(20, 28))},
    ]
    assert required_frame_indices(windows) == {
        "train": list(range(9)),
        "validation": list(range(20, 28)),
    }


def test_cls_and_patch_mean_make_one_ordered_feature_per_frame():
    tokens = np.zeros((2, 5, 3), dtype=np.float32)
    tokens[:, 0, :] = [1, 2, 3]
    tokens[:, 1:, :] = [4, 6, 8]
    features = cls_plus_patch_mean(tokens)
    assert features.shape == (2, 6)
    assert features[0].tolist() == [1, 2, 3, 4, 6, 8]
