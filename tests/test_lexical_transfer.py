from scripts.build_conditional_curriculum import build, verify
from scripts.build_lexical_range_curriculum import reword, template
from scripts.lexical_transfer import CHECKPOINTS, transform_new
from scripts.range_generalization import VARIANTS, transform


def test_novel_templates_and_preserved_targets():
    rows = build(127, 4, "development")
    previous = {template(reword(r, v)["question"]) for r in rows for v in range(4)}
    previous |= {template(transform(r, v)["question"]) for r in rows for v in VARIANTS}
    for row in rows:
        for variant in VARIANTS:
            actual = transform_new(row, variant)
            verify(actual)
            assert actual["options"] == row["options"]
            assert actual["target"] == row["target"]
            assert actual == transform_new(row, variant)
            if variant in ("wording", "combined"):
                assert template(actual["question"]) not in previous
    assert CHECKPOINTS["parent"][1].startswith("f287bb")
    assert CHECKPOINTS["candidate"][1].startswith("fccee2")
