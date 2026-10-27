import pytest

pytest.importorskip("torch")
from scripts.audit_openimages_existing_train_labels import inventory  # noqa: E402


def test_inventory_requires_explicit_negatives_and_matching_feature():
    rows = [{"image_id": "1", "split": "train", "target_label": "a"}]
    groups = [{"image_id": "1", "positive_labels": {"a"},
               "negative_labels": {"b", "c", "d", "outside"}}]
    result = inventory(groups, rows, {"1"}, set("abcd"))
    assert result[0]["eligible"] and result[0]["feature_available"]
    assert result[0]["negative_labels"] == ["b", "c", "d"]


def test_inventory_rejects_missing_target_or_image():
    rows = [{"image_id": "1", "split": "train", "target_label": "a"}]
    with pytest.raises(ValueError, match="positive"):
        inventory([{"image_id": "1", "positive_labels": set(), "negative_labels": set()}],
                  rows, set(), {"a"})
    with pytest.raises(ValueError, match="not all"):
        inventory([], rows, set(), {"a"})


def test_inventory_rejects_validation_rows():
    rows = [{"image_id": "1", "split": "validation", "target_label": "a"}]
    with pytest.raises(ValueError, match="train-only"):
        inventory([], rows, set(), {"a"})
