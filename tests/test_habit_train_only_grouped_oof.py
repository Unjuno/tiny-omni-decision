from __future__ import annotations

import pytest
import torch

from scripts.run_habit_train_only_grouped_oof import (
    assign_folds,
    connected_components,
    evaluate_oof,
)


def _example(index: int, task: str, video: str) -> dict:
    return {
        "id": f"train:{index}",
        "sid": task,
        "video_sha256": video,
        "target_index": index % 2,
        "options": ["a", "b"],
    }


def test_components_join_rows_sharing_either_task_or_video() -> None:
    rows = [
        _example(0, "task-a", "video-1"),
        _example(1, "task-a", "video-2"),
        _example(2, "task-b", "video-2"),
        _example(3, "task-c", "video-3"),
    ]

    assert connected_components(rows, list(range(4))) == [[0, 1, 2], [3]]


def test_fold_assignment_is_deterministic_complete_and_leak_free() -> None:
    rows = [
        _example(i, f"task-{i // 2}", f"video-{i // 2}")
        for i in range(14)
    ]

    folds = assign_folds(rows, list(range(len(rows))), fold_count=4)

    assert folds == assign_folds(rows, list(range(len(rows))), fold_count=4)
    assert sorted(index for fold in folds for index in fold) == list(range(len(rows)))
    assert all(folds)
    for heldout in folds:
        train = set(range(len(rows))) - set(heldout)
        assert {rows[i]["sid"] for i in heldout}.isdisjoint(
            {rows[i]["sid"] for i in train}
        )
        assert {rows[i]["video_sha256"] for i in heldout}.isdisjoint(
            {rows[i]["video_sha256"] for i in train}
        )


def test_oof_metric_aggregation_requires_exact_train_row_coverage() -> None:
    rows = [
        _example(0, "task-a", "video-1"),
        _example(1, "task-b", "video-2"),
    ]
    predictions = {0: torch.tensor([2.0, 0.0]), 1: torch.tensor([0.0, 2.0])}

    report = evaluate_oof(predictions, [0, 1], rows)

    assert report["count"] == 2
    assert report["accuracy"] == 1.0
    assert [row["sample_id"] for row in report["predictions"]] == ["train:0", "train:1"]
    with pytest.raises(ValueError, match="exactly once"):
        evaluate_oof({0: predictions[0]}, [0, 1], rows)
