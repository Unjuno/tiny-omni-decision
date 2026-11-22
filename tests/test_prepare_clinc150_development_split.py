import pytest

from scripts.prepare_clinc150_development_split import build_grouped_split


def _rows():
    rows = []
    for label in range(151):
        for index in range(5):
            rows.append({"text": f"intent {label} example {index}", "intent": label})
    # Exact duplicate utterance with conflicting labels must stay in one group.
    rows.extend(
        [
            {"text": "same normalized phrase", "intent": 0},
            {"text": " SAME   normalized phrase ", "intent": 1},
        ]
    )
    return rows


def test_split_is_deterministic_and_disjoint_by_normalized_content():
    names = [f"class-{index}" for index in range(151)]
    first = build_grouped_split(_rows(), names)
    second = build_grouped_split(_rows(), names)
    assert first == second
    train, validation, receipt = first
    train_groups = {row["content_group_id"] for row in train}
    validation_groups = {row["content_group_id"] for row in validation}
    assert not train_groups & validation_groups
    assert len(train) + len(validation) == len(_rows())
    assert receipt["conflicting_label_content_groups"] == 1
    assert all(receipt["train_class_counts"][str(i)] for i in range(151))
    assert all(receipt["validation_class_counts"][str(i)] for i in range(151))


def test_split_rejects_missing_classes_and_bad_rows():
    with pytest.raises(ValueError, match="151 unique class names"):
        build_grouped_split(_rows(), ["too-few"])
    with pytest.raises(ValueError, match="nonempty strings"):
        build_grouped_split(_rows() + [{"text": " ", "intent": 0}], [f"c{i}" for i in range(151)])
