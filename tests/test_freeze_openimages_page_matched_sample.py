import pytest

from scripts.freeze_openimages_page_matched_sample import build_page_matched_pair


def _manifest(split, ids):
    return {
        "split": split,
        "target_labels": ["/cat"],
        "examples": [
            {"sample_id": f"{split}-{image_id}", "image_id": image_id, "target_label": "/cat"}
            for image_id in ids
        ],
    }


def _receipt(image_id, split, md5, page_match=True):
    return {
        "ImageID": image_id,
        "Subset": split,
        "OriginalMD5": md5,
        "page_license_and_attribution_match": page_match,
    }


def test_freezes_only_page_matched_examples_and_checks_split_integrity():
    train, validation, summary = build_page_matched_pair(
        _manifest("train", ["a", "b"]),
        _manifest("validation", ["c", "d"]),
        [
            _receipt("a", "train", "md5-a"),
            _receipt("b", "train", "md5-b", False),
        ],
        [
            _receipt("c", "validation", "md5-c"),
            _receipt("d", "validation", "md5-d", False),
        ],
        train_receipt_sha256="a" * 64,
        validation_receipt_sha256="b" * 64,
    )

    assert [row["image_id"] for row in train["examples"]] == ["a"]
    assert [row["image_id"] for row in validation["examples"]] == ["c"]
    assert train["individual_rights_verified"] is False
    assert train["excluded_nonmatching_or_unavailable_count"] == 1
    assert summary["train_validation_image_id_overlap"] == 0
    assert summary["train_validation_original_md5_overlap"] == 0


@pytest.mark.parametrize(
    "train_rows,validation_rows,error",
    [
        (
            [_receipt("same", "train", "md5-a")],
            [_receipt("same", "validation", "md5-b")],
            "ImageID overlap",
        ),
        (
            [_receipt("a", "train", "md5-same")],
            [_receipt("b", "validation", "md5-same")],
            "OriginalMD5 overlap",
        ),
    ],
)
def test_rejects_train_validation_identity_overlap(train_rows, validation_rows, error):
    with pytest.raises(ValueError, match=error):
        build_page_matched_pair(
            _manifest("train", [train_rows[0]["ImageID"]]),
            _manifest("validation", [validation_rows[0]["ImageID"]]),
            train_rows,
            validation_rows,
            train_receipt_sha256="a" * 64,
            validation_receipt_sha256="b" * 64,
        )
