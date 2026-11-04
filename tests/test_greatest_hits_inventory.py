import hashlib

import pytest


def fixture_ids():
    train = [f"2015-03-{day:02d}-12-00-00-{i}" for day in range(1, 9) for i in range(2)]
    test = ["2015-03-01-12-00-00-2", "2015-04-01-12-00-00"]
    return train, test


def test_partition_is_fixed_before_labels_and_keeps_date_groups():
    from scripts.greatest_hits_inventory import partition_train

    train, test = fixture_ids()
    plan = partition_train(train, test)
    assert plan == partition_train(train[::-1], test[::-1])
    assert plan["quarantine"] == sorted(train[:2])
    expected_dates = sorted({i[:10] for i in train[2:]},
                            key=lambda s: hashlib.sha256(("17:" + s).encode()).hexdigest())[:5]
    assert {i[:10] for i in plan["validation"]} == set(expected_dates)
    assert not set(plan["train"]) & set(plan["validation"])
    assert not {i[:10] for i in plan["train"]} & {i[:10] for i in plan["validation"]}
    assert sorted(sum(plan.values(), [])) == sorted(train)


@pytest.mark.parametrize("kind", ["duplicate", "overlap", "format", "too_few_dates"])
def test_bad_partition_input_fails(kind):
    from scripts.greatest_hits_inventory import partition_train

    train, test = fixture_ids()
    if kind == "duplicate":
        train += train[:1]
    elif kind == "overlap":
        test += train[:1]
    elif kind == "format":
        train[0] = "../../unsafe"
    else:
        train = train[:6]
    with pytest.raises(ValueError):
        partition_train(train, test)


def test_annotation_inventory_keeps_missing_unknown_and_unsupported_classes():
    from scripts.greatest_hits_inventory import account_annotations

    plan = {"train": ["a"], "validation": ["b"], "quarantine": ["c"]}
    annotations = {
        "a": [dict(seconds=1, fields=["wood", "hit", None]),
              dict(seconds=2, fields=[None, None, None])],
        "b": [dict(seconds=1, fields=["unreviewed", "hit", "static"])],
        "c": [dict(seconds=1, fields=["metal", "scratch", "static"])],
    }
    result = account_annotations(plan, annotations)
    assert result["splits"]["train"]["rows"] == 2
    assert result["splits"]["train"]["fields"][0] == {"wood": 1, "<MISSING>": 1}
    assert result["splits"]["validation"]["fields"][0] == {"unreviewed": 1}
    assert result["validation_values_absent_from_train"] == [["unreviewed"], [], ["static"]]
    assert result["task_vocabularies"][0] == ["metal", "unreviewed", "wood"]
    assert result["task_vocabularies"][2] == ["static"]
    with pytest.raises(ValueError, match="exact"):
        account_annotations(plan, {k: v for k, v in annotations.items() if k != "c"})
