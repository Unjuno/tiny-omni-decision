from scripts.build_banking77_train_validation_split import (
    EXPECTED_REVISION,
    EXPECTED_TRAIN_SHA256,
    build_split_manifest,
)


def _rows():
    rows = []
    for label in range(3):
        for sample in range(10):
            rows.append({"text": f"Label {label} example {sample}", "label": label})
    rows.extend(
        [
            {"text": "  LABEL 0 EXAMPLE 0  ", "label": 0},
            {"text": "Label 1 example 2", "label": 1},
        ]
    )
    return rows


def _build(rows=None, **kwargs):
    return build_split_manifest(
        _rows() if rows is None else rows,
        source_revision=EXPECTED_REVISION,
        source_sha256=EXPECTED_TRAIN_SHA256,
        **kwargs,
    )


def test_split_is_deterministic_and_content_group_disjoint():
    first = _build()
    second = _build()

    assert first == second
    assert first["overlap"] == {"row_indices": 0, "normalized_content_groups": 0}
    assert first["train"]["row_count"] + first["validation"]["row_count"] == len(_rows())
    assert first["train"]["class_counts"].keys() == first["validation"]["class_counts"].keys()
    assert all(count > 0 for count in first["validation"]["class_counts"].values())


def test_normalized_duplicates_are_assigned_together():
    rows = _rows()
    manifest = _build(rows)
    train = set(manifest["train"]["row_indices"])
    validation = set(manifest["validation"]["row_indices"])

    assert (0 in train and 30 in train) or (0 in validation and 30 in validation)
    assert (12 in train and 31 in train) or (12 in validation and 31 in validation)


def test_conflicting_labels_for_normalized_duplicate_fail_closed():
    rows = _rows()
    rows.append({"text": "label 0 example 0", "label": 2})

    try:
        _build(rows)
    except ValueError as exc:
        assert "conflicting labels" in str(exc)
    else:
        raise AssertionError("conflicting normalized duplicates must be rejected")


def test_wrong_source_revision_or_hash_fails_closed():
    import pytest

    with pytest.raises(ValueError, match="revision"):
        build_split_manifest(
            _rows(), source_revision="main", source_sha256=EXPECTED_TRAIN_SHA256
        )
    with pytest.raises(ValueError, match="hash"):
        build_split_manifest(
            _rows(), source_revision=EXPECTED_REVISION, source_sha256="0" * 64
        )
