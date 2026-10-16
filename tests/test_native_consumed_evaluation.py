import numpy as np
import pytest


def rows():
    return [
        dict(
            id=str(i),
            split="train",
            source="s",
            task="t",
            modality="text",
            observation=str(i),
            group=str(i),
            options=["a", "b"],
            target_index=i % 2,
        )
        for i in range(4)
    ]


def test_complete_consumption_not_probe_or_last_step_and_multiplicity():
    from scripts.native_consumed_evaluation import consumed_plan

    history = [dict(step=1, sample_ids=["3", "1"]), dict(step=2, sample_ids=["1", "0"])]
    plan = consumed_plan(rows(), {"one": history, "two": history[::-1][::-1]})
    assert plan["ids"] == ["0", "1", "3"]
    assert plan["multiplicities"] == [1, 2, 1]
    assert plan["occurrence_count"] == 4
    assert plan["unique_count"] == 3
    assert plan["unconsumed_count"] == 1


@pytest.mark.parametrize("case", ["missing", "different", "steps", "split", "duplicate"])
def test_invalid_or_nonmatching_consumption_rejected(case):
    from scripts.native_consumed_evaluation import consumed_plan

    data = rows()
    first = [dict(step=1, sample_ids=["0", "1"])]
    other = [dict(step=1, sample_ids=["0", "1"])]
    if case == "missing":
        first[0]["sample_ids"] = ["unknown"]
    elif case == "different":
        other[0]["sample_ids"] = ["0", "0"]
    elif case == "steps":
        first[0]["step"] = 2
    elif case == "split":
        data[0]["split"] = "validation"
    else:
        data.append(data[0])
    with pytest.raises(ValueError):
        consumed_plan(data, {"one": first, "two": other})


def test_gap_uses_only_exact_common_tasks_and_records_missing_tasks():
    from scripts.native_consumed_evaluation import common_task_gap

    train = rows()[:2] + [dict(rows()[2], task="train-only")]
    validation = [
        dict(rows()[0], id="v0", split="validation"),
        dict(rows()[1], id="v1", split="validation", task="validation-only"),
    ]
    scores = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    result = common_task_gap(train, scores, validation, np.array([[0.0, 3.0], [0.0, 3.0]]))
    assert result["common_tasks"] == ["s:t"]
    assert result["train_only_tasks"] == ["s:train-only"]
    assert result["validation_only_tasks"] == ["s:validation-only"]
    assert result["train"]["macro"]["accuracy"] == 1
    assert result["validation"]["macro"]["accuracy"] == 0
    assert result["validation_minus_train"]["accuracy"] == -1


def test_common_gap_rejects_split_or_id_overlap():
    from scripts.native_consumed_evaluation import common_task_gap

    train = rows()[:2]
    score = np.array([[3.0, 0.0], [0.0, 3.0]])
    with pytest.raises(ValueError):
        common_task_gap(train, score, train, score)
    with pytest.raises(ValueError):
        common_task_gap(train, score, [dict(r, split="validation") for r in train], score)
