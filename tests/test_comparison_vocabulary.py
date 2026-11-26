import copy

from scripts.build_comparison_vocabulary import WORDS, expand
from scripts.build_conditional_curriculum import build
from scripts.build_lexical_range_curriculum import build_rows
from scripts.build_numeric_range_curriculum import apply_by_group


def test_vocabulary_changes_only_operator_preserving_opposite_pairs():
    rows = build_rows(apply_by_group(build(17, 64, "train"), [(1, 0), (2, 7)], 71))
    saved = copy.deepcopy(rows)
    actual = expand(rows)
    assert actual == expand(rows)
    assert saved == rows
    by_group = {}
    for a, b in zip(rows, actual, strict=True):
        assert {k: v for k, v in a.items() if k != "question"} == {
            k: v for k, v in b.items() if k not in ("question", "comparison_variant")
        }
        if b["comparison_variant"] is None:
            assert a["question"] == b["question"]
        else:
            direction = a["oracle"]["direction"]
            word = WORDS[b["comparison_variant"]][int(direction == "maximum")]
            assert a["question"].replace(direction, word) == b["question"]
            by_group.setdefault(b["group"], set()).add(b["comparison_variant"])
    assert all(len(v) == 1 for v in by_group.values())
