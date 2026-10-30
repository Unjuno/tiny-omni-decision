import numpy as np
import pytest

from tiny_omni_decision.eidon_pairing import (
    build_four_choice_queries,
    relative_orientation_window_features,
)


def _group(contributor="person-1", task="laundry", split="train"):
    return [
        {
            "recording_id": f"{contributor}-{split}-recording-{index}",
            "contributor_id": contributor,
            "task_type": task,
            "split": split,
            "asset_sha256": f"{contributor}-{split}-asset-hash-{index}",
        }
        for index in range(4)
    ]


def _orientation_rows(*, missing=None, chest_angle=0.0):
    timestamps = []
    slots = []
    quaternions_xyzw = []
    for time_ms in range(0, 8000, 250):
        for slot in range(7):
            if missing == (time_ms, slot):
                continue
            angle = chest_angle if slot == 6 else 0.0
            timestamps.append(time_ms)
            slots.append(slot)
            quaternions_xyzw.append(
                [0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]
            )
    return timestamps, slots, quaternions_xyzw


def test_four_choice_queries_are_grouped_and_seed_deterministic():
    rows = _group() + _group("person-2", split="validation")

    first = build_four_choice_queries(rows, seed=17)
    replay = build_four_choice_queries(rows, seed=17)

    assert first == replay
    assert len(first) == 8
    for query in first:
        assert len(query["options"]) == 4
        assert len(set(query["options"])) == 4
        assert query["options"][query["target_index"]] == query["query_recording_id"]
        assert query["contributor_id"] in {"person-1", "person-2"}


def test_four_choice_queries_reject_contributor_cross_split_leakage():
    rows = _group(split="train") + _group(split="validation")

    with pytest.raises(ValueError, match="contributor appears in multiple splits"):
        build_four_choice_queries(rows)


def test_four_choice_queries_reject_groups_without_four_recordings():
    with pytest.raises(ValueError, match="exactly four unique recordings"):
        build_four_choice_queries(_group()[:3])


def test_four_choice_queries_reject_asset_hash_overlap_between_splits():
    train = _group(split="train")
    validation = _group("person-2", split="validation")
    validation[0]["asset_sha256"] = train[0]["asset_sha256"]

    with pytest.raises(ValueError, match="asset hash appears in multiple splits"):
        build_four_choice_queries(train + validation)


def test_relative_orientation_features_are_chest_relative_and_binned():
    times, slots, quaternions = _orientation_rows(chest_angle=np.pi / 2)

    result = relative_orientation_window_features(
        times,
        slots,
        quaternions,
        window_start_ms=0,
        window_end_ms=8000,
        bins=8,
    )

    expected_relative_z_rotation = np.array([0, 1, 0, -1, 0, 0, 0, 0, 1])
    assert result.shape == (8, 54)
    np.testing.assert_allclose(
        result[:, :9], np.broadcast_to(expected_relative_z_rotation, (8, 9)), atol=1e-6
    )
    assert np.isfinite(result).all()


def test_relative_orientation_features_reject_missing_sensor_slots():
    times, slots, quaternions = _orientation_rows(missing=(1000, 3))

    with pytest.raises(ValueError, match="every timestamp must contain slots 0 through 6"):
        relative_orientation_window_features(
            times,
            slots,
            quaternions,
            window_start_ms=0,
            window_end_ms=8000,
            bins=8,
        )
