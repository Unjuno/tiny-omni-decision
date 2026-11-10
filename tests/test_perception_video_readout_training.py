from scripts.train_perception_test_video_readout import (
    selection_improved,
    should_stop,
    split_training_groups,
)


def test_internal_selection_split_is_deterministic_and_video_grouped():
    rows = [
        {"id": f"q-{video}-{question}", "source_video_id": f"video-{video}"}
        for video in range(20)
        for question in range(video % 4 + 1)
    ]

    train, selection, metadata = split_training_groups(rows, seed=17)
    train_again, selection_again, metadata_again = split_training_groups(rows, seed=17)
    train_groups = {rows[index]["source_video_id"] for index in train}
    selection_groups = {rows[index]["source_video_id"] for index in selection}

    assert train == train_again
    assert selection == selection_again
    assert metadata == metadata_again
    assert train_groups.isdisjoint(selection_groups)
    assert train_groups | selection_groups == {row["source_video_id"] for row in rows}
    assert metadata["selection_group_count"] == 3


def test_internal_split_rejects_single_video_group():
    rows = [{"id": "q1", "source_video_id": "same"}, {"id": "q2", "source_video_id": "same"}]

    try:
        split_training_groups(rows)
    except ValueError as error:
        assert "two video groups" in str(error)
    else:
        raise AssertionError("single-group split unexpectedly succeeded")


def test_selection_rule_uses_nll_then_brier_with_declared_tolerance():
    assert selection_improved(0.9, 0.5, 1.0, 0.6)
    assert selection_improved(1.0 + 5e-6, 0.4, 1.0, 0.5)
    assert not selection_improved(1.0 + 2e-5, 0.1, 1.0, 0.5)
    assert not selection_improved(1.0, 0.6, 1.0, 0.5)


def test_early_stopping_patience_is_exactly_six_evaluated_epochs():
    assert not should_stop(5)
    assert should_stop(6)
