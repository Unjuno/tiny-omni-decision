import copy
from types import SimpleNamespace

import numpy as np
import pytest


def rows():
    return [
        dict(
            id=f"{task}:{date}:{i}",
            parent=f"{date}-{i}",
            date_group=date,
            event_id=f"{date}-{i}",
            asset_sha256=f"{date}-{i}",
            split="validation",
            task=task,
            question=task,
            options=["a", "b"],
            target=i % 2,
        )
        for task in ("action", "material", "reaction")
        for date in ("date1", "date2", "date3")
        for i in range(3)
    ]


def test_plan_deterministic_label_blind_and_complete():
    from scripts.greatest_hits_replacement import build_inventory

    data = rows()
    original = copy.deepcopy(data)
    plan = build_inventory(data)
    assert len(plan["pairs"]) == len(data)
    assert build_inventory(list(reversed(data))) == plan
    for r in data:
        r["target"] = 1 - r["target"]
    assert build_inventory(data) == plan
    assert original != data
    by_id = {r["id"]: r for r in data}
    for pair in plan["pairs"]:
        a, b = by_id[pair["id"]], by_id[pair["donor_id"]]
        assert a["date_group"] != b["date_group"]
        assert a["parent"] != b["parent"] and a["asset_sha256"] != b["asset_sha256"]
        assert a["task"] == b["task"]
        assert a["options"] == b["options"] and a["question"] == b["question"]


@pytest.mark.parametrize("change", ["one_date", "train", "duplicate", "options", "question"])
def test_plan_fails_instead_of_dropping_or_redrawing(change):
    from scripts.greatest_hits_replacement import build_inventory

    data = rows()
    if change == "one_date":
        data = [r for r in data if r["date_group"] == "date1"]
    if change == "train":
        data[0]["split"] = "train"
    if change == "duplicate":
        data.append(data[0])
    if change == "options":
        data[0]["options"].reverse()
    if change == "question":
        data[0]["question"] = "changed"
    with pytest.raises(ValueError):
        build_inventory(data)


def test_replacement_keeps_original_query_and_exact_physical_blocks():
    from scripts.greatest_hits_replacement import inputs

    base = [SimpleNamespace(kind=k, marker="base") for k in ("speech", "event_audio", "video")]
    donor = [SimpleNamespace(kind=k, marker="donor") for k in ("speech", "event_audio", "video")]
    question = object()

    class Store:
        binding = {
            "questions": {
                i: dict(options=["a", "b"], query_keys=["x", "y"]) for i in ("base", "donor")
            }
        }

        def get(self, ident):
            return (base, question) if ident == "base" else (donor, object())

    for mode, expected in [
        ("audio", [donor[0], donor[1], base[2]]),
        ("video", [base[0], base[1], donor[2]]),
    ]:
        blocks, q = inputs(Store(), "base", "donor", mode)
        assert all(a is b for a, b in zip(blocks, expected, strict=True))
        assert q is question
        blocks, q = inputs(Store(), "base", "base", mode)
        assert all(a is b for a, b in zip(blocks, base, strict=True))
        assert q is question
    with pytest.raises(ValueError):
        inputs(Store(), "base", "donor", "unknown")


def test_probability_sensitivity_has_no_target_accuracy_and_ignores_logit_offset():
    from scripts.greatest_hits_replacement import sensitivity

    original = np.array([[4.0, 0.0], [0.0, 4.0]])
    equal = sensitivity(original, original + 7)
    assert equal["prediction_flip_rate"] == 0
    assert equal["mean_total_variation"] == pytest.approx(0.0, abs=1e-15)
    assert equal["mean_js_divergence"] == pytest.approx(0.0, abs=1e-15)
    changed = sensitivity(original, original[:, ::-1])
    assert changed["prediction_flip_rate"] == 1
    assert changed["mean_total_variation"] > 0.9
    assert "accuracy" not in changed and "nll" not in changed
    with pytest.raises(ValueError):
        sensitivity(original, np.full_like(original, np.nan))


@pytest.mark.parametrize("shape", [(0, 2), (2, 2, 2), (2,), (2, 1)])
def test_sensitivity_requires_nonempty_question_by_option_matrices(shape):
    from scripts.greatest_hits_replacement import sensitivity

    with pytest.raises(ValueError):
        sensitivity(np.zeros(shape), np.zeros(shape))
