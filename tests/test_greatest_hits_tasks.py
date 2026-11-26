import pytest


def test_each_known_field_produces_full_vocabulary_task_without_label_in_prompt():
    from scripts.greatest_hits_tasks import tasks_for_parent

    rows = [dict(seconds=1.0, fields=["wood", "hit", "static"]),
            dict(seconds=2.0, fields=[None, "scratch", None])]
    tasks, missing = tasks_for_parent("parent", "train", "a" * 64, rows, set())
    assert len(tasks) == 4 and missing == {"material": 1, "action": 0, "reaction": 1}
    assert [len(t["options"]) for t in tasks] == [17, 2, 5, 2]
    assert tasks[0]["options"][tasks[0]["target"]] == "wood"
    assert "wood" not in tasks[0]["question"].lower()
    assert tasks[0]["event_window_seconds"] == [0.75, 1.25]
    assert tasks[0]["asset_sha256"] == "a" * 64
    assert len({t["id"] for t in tasks}) == 4
    assert tasks[0]["event_id"] == tasks[1]["event_id"]


def test_invalid_windows_or_unknown_categories_are_not_silently_dropped():
    from scripts.greatest_hits_tasks import tasks_for_parent

    rows = [dict(seconds=1.0, fields=["wood", "hit", "static"])]
    with pytest.raises(ValueError, match="window"):
        tasks_for_parent("parent", "validation", "a" * 64, rows, {0})
    rows[0]["fields"][0] = "unknown-material"
    with pytest.raises(ValueError, match="category"):
        tasks_for_parent("parent", "validation", "a" * 64, rows, set())


def test_validation_prior_is_fitted_only_to_training_labels():
    from scripts.greatest_hits_tasks import prior_metrics

    result = prior_metrics([3, 1], [1, 3])
    assert result["probabilities"] == [0.75, 0.25]
    assert result["count"] == 4 and result["accuracy"] == 0.25
    assert result["brier"] == 0.875
    assert result["ece"] == 0.5
