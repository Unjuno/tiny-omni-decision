import copy

from scripts.build_comparison_vocabulary import expand
from scripts.build_conditional_curriculum import build
from scripts.build_lexical_range_curriculum import build_rows
from scripts.build_paired_wording import views


def test_four_views_preserve_state_options_and_opposing_targets():
    parents = build(17, 4, "train")
    for p in parents:
        p["numeric_transform"] = {"scale": 1, "offset": 0}
    parents = expand(build_rows(parents))
    original = copy.deepcopy(parents)
    for parent in parents:
        if parent["task"] not in ("extremum", "temporal_order"):
            continue
        rows = views(parent)
        assert rows == views(parent)
        assert len({r["question"] for r in rows}) == 4
        assert len({r["id"] for r in rows}) == 4
        assert rows[0]["question"] == parent["question"]
        for row in rows:
            assert row["comparison_variant"] == (parent["comparison_variant"] + row["view"]) % 4
            assert row["target"] == parent["target"]
            assert row["state"] == parent["state"]
            assert row["options"] == parent["options"]
    assert parents == original
