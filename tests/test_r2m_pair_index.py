import copy

import pytest

from tiny_omni_decision.r2m_pairs import build_audio_vibration_pairs


def records():
    return [
        {"activity": "01_walk", "participant_id": 2, "trial_id": 10,
         "split": "train", "audio_samples": 16000},
        {"activity": "01_walk", "participant_id": 4, "trial_id": 20,
         "split": "train", "audio_samples": 16800},
        {"activity": "01_walk", "participant_id": 4, "trial_id": 21,
         "split": "train", "audio_samples": 32000},
        {"activity": "01_walk", "participant_id": 9, "trial_id": 30,
         "split": "validation", "audio_samples": 16000},
        {"activity": "01_walk", "participant_id": 15, "trial_id": 40,
         "split": "validation", "audio_samples": 15900},
    ]


def test_pair_index_is_deterministic_and_prefers_duration_matched_other_person():
    source = records()
    first = build_audio_vibration_pairs(source)
    second = build_audio_vibration_pairs(reversed(source))
    assert first == second
    train_pairs = [row for row in first["pairs"] if row["split"] == "train"]
    assert len(train_pairs) == 2
    assert all(row["negative_participant_relation"] == "cross_participant"
               for row in train_pairs)
    assert all(row["negative_duration_delta_fraction"] <= 0.10
               for row in train_pairs)
    assert first["sealed_audit_accessed"] is False


def test_pair_index_rejects_sealed_quarantined_and_cross_split_participants():
    sealed = records() + [{"activity": "01_walk", "participant_id": 1,
                           "trial_id": 1, "split": "sealed_audit",
                           "audio_samples": 16000}]
    with pytest.raises(ValueError, match="outside the fitting scope"):
        build_audio_vibration_pairs(sealed)

    cross_split = records() + [{"activity": "02_open", "participant_id": 2,
                                "trial_id": 50, "split": "validation",
                                "audio_samples": 16000}]
    with pytest.raises(ValueError, match="multiple splits"):
        build_audio_vibration_pairs(cross_split)


def test_pair_index_emits_no_negative_when_duration_gate_has_no_match():
    rows = records()
    rows[1]["audio_samples"] = 100000
    rows[2]["audio_samples"] = 200000
    index = build_audio_vibration_pairs(rows, max_duration_delta=0.01)
    assert not [row for row in index["pairs"] if row["split"] == "train"]
    assert index["unpaired_trial_count_by_split"]["train"] == 3


def test_pair_index_rejects_duplicate_trial_and_bad_seed():
    rows = records()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate source trial"):
        build_audio_vibration_pairs(rows)
    with pytest.raises(ValueError, match="seed"):
        build_audio_vibration_pairs(records(), seed=-1)
