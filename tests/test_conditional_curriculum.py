import pytest

from scripts.build_conditional_curriculum import build, verify


def test_curriculum_determinism_grouping_and_oracle():
    rows = build(17, 32, "train")
    other = build(29, 32, "validation")
    assert rows == build(17, 32, "train")
    assert len(rows) == 256
    assert not {r["group"] for r in rows} & {r["group"] for r in other}
    assert len({r["id"] for r in rows}) == len(rows)
    for row in rows + other:
        verify(row)
        assert len(set(row["options"])) == len(row["options"])
        wrong = next(v for v in row["options"] if v != row["target"])
        with pytest.raises(ValueError):
            verify(dict(row, target=wrong))
