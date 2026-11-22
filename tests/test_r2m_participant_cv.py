import pytest

from tiny_omni_decision.r2m_participant_cv import leave_one_participant_out_plan


def _rows():
    return [
        {"sample_id": f"p{participant}-{label}", "participant_id": participant,
         "activity": label}
        for participant in (2, 4, 7, 13)
        for label in ("walk", "open", "close")
    ]


def test_lopo_plan_is_deterministic_complete_and_participant_disjoint():
    rows = _rows()
    first = leave_one_participant_out_plan(rows, seed=17)
    second = leave_one_participant_out_plan(rows, seed=17)

    assert first == second
    assert len(first) == 4
    seen_validation = set()
    for fold in first:
        train = set(fold["train_sample_ids"])
        calibration = set(fold["calibration_sample_ids"])
        validation = set(fold["validation_sample_ids"])
        assert not train & calibration
        assert not train & validation
        assert not calibration & validation
        assert len(validation) == 3
        assert fold["validation_participant"] not in fold["train_participants"]
        assert fold["validation_participant"] != fold["calibration_participant"]
        assert not seen_validation & validation
        seen_validation |= validation
    assert seen_validation == {row["sample_id"] for row in rows}


def test_lopo_plan_fails_when_no_single_participant_can_calibrate_all_labels():
    rows = [
        {"sample_id": "p1-a", "participant_id": 1, "activity": "a"},
        {"sample_id": "p2-b", "participant_id": 2, "activity": "b"},
        {"sample_id": "p3-a", "participant_id": 3, "activity": "a"},
    ]

    with pytest.raises(ValueError, match="no participant-only calibration group"):
        leave_one_participant_out_plan(rows)


def test_lopo_plan_rejects_duplicate_sample_ids():
    rows = _rows()
    rows[1] = {**rows[1], "sample_id": rows[0]["sample_id"]}

    with pytest.raises(ValueError, match="duplicate sample ID"):
        leave_one_participant_out_plan(rows)
