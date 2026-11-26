import pytest

from scripts.build_conditional_curriculum import build, verify
from scripts.comparative_word_diagnostic import VARIANTS, render
from scripts.range_generalization import validate_evaluation_rows


def test_factorial_preserves_answers_and_pairs():
    base = [r for r in build(137, 4, "development") if r["task"] == "extremum"]
    rows = [render(r, v) for v in VARIANTS for r in base]
    validate_evaluation_rows(rows, VARIANTS, 128)
    for r in base:
        for variant in VARIANTS:
            actual = render(r, variant)
            verify(actual)
            assert actual["options"] == r["options"] and actual["target"] == r["target"]
    a = render(base[0], "base-minimum-select")
    b = render(base[0], "base-smallest-select")
    assert a["state"] == b["state"]
    assert a["question"].replace("minimum", "smallest") == b["question"]
    with pytest.raises(ValueError):
        validate_evaluation_rows(rows[:-1], VARIANTS, 128)
    with pytest.raises(ValueError):
        validate_evaluation_rows(rows, VARIANTS[:-1], 128)
