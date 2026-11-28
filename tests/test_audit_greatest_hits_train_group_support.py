import pytest

from scripts.audit_greatest_hits_train_group_support import summarize_train_group_support


def _inputs():
    inventory = {
        "source_commit": "a" * 40,
        "partition_sha256": "b" * 64,
        "partition_hashes": {"train": "c" * 64},
        "splits": {
            "train": {
                "parents": 2,
                "rows": 3,
                "complete": 2,
                "missing_fields": {"0": 2, "1": 1},
                "fields": [
                    {"wood": 2, "metal": 1, "<MISSING>": 0},
                    {"hit": 2, "scratch": 0, "<MISSING>": 1},
                    {"static": 2, "splash": 1, "<MISSING>": 0},
                ],
            }
        },
        "parents": {
            "train-a": {
                "fields": [
                    {"wood": 2, "metal": 0, "<MISSING>": 0},
                    {"hit": 1, "scratch": 0, "<MISSING>": 1},
                    {"static": 1, "splash": 1, "<MISSING>": 0},
                ]
            },
            "train-b": {
                "fields": [
                    {"wood": 0, "metal": 1, "<MISSING>": 0},
                    {"hit": 1, "scratch": 0, "<MISSING>": 0},
                    {"static": 1, "splash": 0, "<MISSING>": 0},
                ]
            },
            "validation-only": {
                "fields": [
                    {"wood": 0, "metal": 100, "<MISSING>": 0},
                    {"hit": 0, "scratch": 100, "<MISSING>": 0},
                    {"static": 0, "splash": 100, "<MISSING>": 0},
                ]
            },
        },
    }
    partition = {"train": ["train-a", "train-b"], "validation": ["validation-only"]}
    return inventory, partition


def test_counts_and_parent_support_use_train_groups_only():
    inventory, partition = _inputs()

    report = summarize_train_group_support(inventory, partition)

    assert report["train_parent_count"] == 2
    assert report["tasks"]["material"]["distinct_train_parent_support"] == {
        "wood": 1,
        "metal": 1,
    }
    assert report["tasks"]["reaction"]["distinct_train_parent_support"] == {
        "static": 2,
        "splash": 1,
    }
    assert report["tasks"]["action"]["missing_rows"] == 1
    assert report["tasks"]["action"]["classes_with_fewer_than_10_train_parents"] == [
        "hit",
        "scratch",
    ]


def test_rejects_duplicate_train_parent_ids():
    inventory, partition = _inputs()
    partition["train"] = ["train-a", "train-a"]

    with pytest.raises(ValueError, match="must be unique"):
        summarize_train_group_support(inventory, partition)


def test_rejects_mismatched_parent_level_class_counts():
    inventory, partition = _inputs()
    inventory["parents"]["train-b"]["fields"][0]["wood"] = 1

    with pytest.raises(ValueError, match="do not match TRAIN totals"):
        summarize_train_group_support(inventory, partition)
