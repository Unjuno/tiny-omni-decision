import pytest


def test_validation_alignment_checks_order_and_media_identity():
    from scripts.evaluate_textfree_image_official_validation import align_validation_rows

    labels = [
        {"id": "v1", "split": "val", "color": "red"},
        {"id": "v2", "split": "val", "color": "blue"},
    ]
    manifest = [
        {"id": "v1", "split": "val", "sha256": "a"},
        {"id": "v2", "split": "val", "sha256": "b"},
    ]
    rows = align_validation_rows(["v1", "v2"], ["a", "b"], labels, manifest, ["t1"], ["c"])
    assert [row["id"] for row in rows] == ["v1", "v2"]

    with pytest.raises(ValueError, match="identity/order/hash"):
        align_validation_rows(["v2", "v1"], ["b", "a"], labels, manifest, ["t1"], ["c"])


def test_validation_alignment_rejects_train_overlap_and_duplicate_asset():
    from scripts.evaluate_textfree_image_official_validation import align_validation_rows

    labels = [{"id": "v1", "split": "val", "color": "red"}]
    manifest = [{"id": "v1", "split": "val", "sha256": "a"}]
    with pytest.raises(ValueError, match="overlap"):
        align_validation_rows(["v1"], ["a"], labels, manifest, ["v1"], ["b"])
    with pytest.raises(ValueError, match="unique"):
        align_validation_rows(["v1", "v2"], ["a", "a"], labels * 2, manifest * 2, [], [])
