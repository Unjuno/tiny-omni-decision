import pytest

pytest.importorskip("torch")
from scripts.audit_source_set_errors import error_kind  # noqa: E402


@pytest.mark.parametrize(
    ("target", "selected", "expected"),
    [
        ([], ["none_visible"], ("none", "exact")),
        ([], ["a"], ("none", "false_visible")),
        (["a"], ["none_visible"], ("one", "miss_all_none")),
        (["a", "b"], ["a"], ("multi", "fn_only")),
        (["a"], ["a", "b"], ("one", "fp_only")),
        (["a", "b"], ["a", "c"], ("multi", "fp_fn")),
        (["a", "b"], ["b", "a"], ("multi", "exact")),
    ],
)
def test_error_partition(target, selected, expected):
    assert (
        error_kind(
            {"target_candidate_ids": target, "selected": selected, "candidate_ids": ["a", "b", "c"]}
        )
        == expected
    )


def test_reject_invalid_predictions():
    row = {
        "target_candidate_ids": ["a"],
        "selected": ["none_visible", "a"],
        "candidate_ids": ["a", "b"],
    }
    with pytest.raises(ValueError):
        error_kind(row)
    row["selected"] = ["unknown"]
    with pytest.raises(ValueError):
        error_kind(row)
