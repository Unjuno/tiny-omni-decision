import copy
import json

import pytest

from scripts.build_condition_transfer_controls import VARIANTS, transform
from scripts.build_conditional_curriculum import build, verify


def test_factors_preserve_answers_and_original_rows():
    rows = build(43, 8, "diagnostic")
    original = copy.deepcopy(rows)
    for variant in VARIANTS:
        for row in rows:
            changed = transform(row, variant)
            assert changed == transform(row, variant)
            assert changed["group"] == row["group"]
            assert changed["options"].index(changed["target"]) == row["options"].index(
                row["target"]
            )
            state = json.loads(changed["state"])
            for item in state:
                if "candidate" in item:
                    item["id"] = item.pop("candidate")
                if "price" in item:
                    item["cost"] = item.pop("price")
            changed["state"] = json.dumps(state)
            verify(changed)
            if variant == "control":
                assert changed["question"] == row["question"]
                assert changed["state"] == row["state"]
    assert rows == original
    with pytest.raises(ValueError):
        transform(rows[0], "unknown")


def test_each_factor_changes_only_intended_fields():
    row = build(43, 1, "diagnostic")[0]
    wording = transform(row, "wording")
    assert wording["state"] == row["state"] and wording["options"] == row["options"]
    for variant in ("labels", "identifier_key", "magnitude"):
        assert transform(row, variant)["question"] == row["question"]
    for variant in ("identifier_key", "magnitude", "field_name"):
        assert transform(row, variant)["options"] == row["options"]
