import copy
import math

import numpy as np
import pytest


def fixture():
    from scripts.condition_relevance_corpus import make_rows
    from tiny_omni_decision.dataset import CLEVR4_TAXONOMIES

    state = {k: values[0] for k, values in CLEVR4_TAXONOMIES.items()}
    rows = make_rows(state, "transfer", 0)
    second = dict(state, color=CLEVR4_TAXONOMIES["color"][1])
    rows += make_rows(second, "transfer", 1)
    return rows, {"rows": 128, "states": 2}


def test_transfer_groups_keep_all_cells_and_candidates():
    from scripts.condition_relevance_transfer import validate_transfer_rows

    rows, expected = fixture()
    accounting = validate_transfer_rows(rows, expected, set())
    assert accounting["unique_states"] == 2
    assert accounting["partitions"]["transfer"]["candidate_counts"] == {40: 128}
    assert set(accounting["partitions"]["transfer"]["views"].values()) == {32}


@pytest.mark.parametrize(
    "change",
    [
        "missing_row",
        "duplicate",
        "short_options",
        "wrong_target",
        "overlap",
        "wrong_count",
        "wrong_split",
    ],
)
def test_transfer_integrity_rejects_changed_groups(change):
    from scripts.condition_relevance_transfer import validate_transfer_rows

    rows, expected = fixture()
    reserved = set()
    if change == "missing_row":
        rows.pop()
    if change == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    if change == "short_options":
        rows[0]["options"].pop()
    if change == "wrong_target":
        rows[0]["target"] = "wrong"
    if change == "overlap":
        reserved.add(rows[0]["group"])
    if change == "wrong_count":
        expected["states"] = 256
    if change == "wrong_split":
        rows[0]["split"] = "validation"
    with pytest.raises(ValueError):
        validate_transfer_rows(rows, expected, reserved)


def test_transfer_metrics_report_four_cells_and_actual_query_counts():
    from scripts.condition_relevance_transfer import transfer_metrics

    rows, _ = fixture()
    scores = np.zeros((len(rows), 40))
    for i, row in enumerate(rows):
        scores[i, row["options"].index(row["target"])] = 100.0
    result = transfer_metrics(rows, scores)
    assert result["overall"]["accuracy"] == 1.0
    assert len(result["conditions"]) == 4
    assert all(v["count"] == 32 for v in result["conditions"].values())
    assert all(v["accuracy"] == 1.0 for v in result["condition_fields"].values())
    assert len(result["condition_fields"]) == 16
    assert {v["count"] for v in result["actual_query_slices"].values()} == {1, 2}
    assert result["overall"]["nll"] < 1e-10
    uniform = transfer_metrics(rows, np.zeros_like(scores))
    assert uniform["overall"]["nll"] == pytest.approx(math.log(40))
    assert uniform["overall"]["brier"] == pytest.approx(0.975)


@pytest.mark.parametrize("change", ["nonfinite", "missing_candidate", "misaligned_rows"])
def test_metrics_reject_nonfinite_or_incomplete_scores(change):
    from scripts.condition_relevance_transfer import transfer_metrics

    rows, _ = fixture()
    scores = np.zeros((len(rows), 40))
    if change == "nonfinite":
        scores[0, 0] = np.nan
    if change == "missing_candidate":
        scores = scores[:, :39]
    if change == "misaligned_rows":
        scores = scores[:-1]
    with pytest.raises(ValueError):
        transfer_metrics(rows, scores)
