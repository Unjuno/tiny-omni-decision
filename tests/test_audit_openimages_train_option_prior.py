import pytest

from scripts.audit_openimages_train_option_prior import build_tasks, cross_validate, metrics


def test_tasks_are_deterministic_and_keep_all_images():
    rows = [{"image_id": str(i), "target_label": "a", "positive_labels": ["a"],
             "negative_labels": ["b", "c", "d", "e"]} for i in range(20)]
    saved = {row["image_id"] for row in rows}
    tasks = build_tasks(rows, saved, set("abcde"))
    assert tasks == build_tasks(rows, saved, set("abcde"))
    assert len(tasks) == 20 and all(len(row["options"]) == 4 for row in tasks)
    predictions = cross_validate(tasks)
    assert len(predictions) == 20
    assert all(abs(sum(row["probabilities"]) - 1) < 1e-12 for row in predictions)
    assert metrics(predictions)["accuracy"] == 1.0


def test_tasks_reject_contradictions_and_missing_media():
    row = {"image_id": "1", "target_label": "a", "positive_labels": ["a"],
           "negative_labels": ["a", "b", "c"]}
    with pytest.raises(ValueError, match="positive or negative"):
        build_tasks([row], {"1"}, set("abc"))
    with pytest.raises(ValueError, match="identity"):
        build_tasks([], {"1"}, set("abc"))
