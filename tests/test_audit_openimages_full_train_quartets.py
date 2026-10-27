import pytest

pytest.importorskip("torch")
from scripts.audit_openimages_full_train_quartets import filter_candidates  # noqa: E402


def test_train_id_and_explicit_negative_filter():
    rows = [
        {"image_id": "train", "positive_labels": {"a"}, "negative_labels": {"b", "c", "d"}},
        {"image_id": "other", "positive_labels": {"a"}, "negative_labels": {"b", "c", "d"}},
        {"image_id": "short", "positive_labels": {"a"}, "negative_labels": {"b", "c"}},
    ]
    selected = filter_candidates(rows, {"train", "short"}, set("abcd"))
    assert [row["image_id"] for row in selected] == ["train"]


def test_reject_conflicting_positive_negative():
    rows = [{"image_id": "a", "positive_labels": {"x"}, "negative_labels": {"x"}}]
    with pytest.raises(ValueError, match="conflicting"):
        filter_candidates(rows, {"a"}, {"x"})
