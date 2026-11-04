import copy
import json

import pytest

from scripts.build_conditional_curriculum import build, verify
from scripts.range_generalization import CHECKPOINTS, VARIANTS, reject_overlap, transform


def test_fixed_generation_and_answers():
    base = build(97, 4, "development")
    original = copy.deepcopy(base)
    for variant in VARIANTS:
        rows = [transform(r, variant) for r in base]
        assert rows == [transform(r, variant) for r in build(97, 4, "development")]
        for old, new in zip(base, rows, strict=True):
            verify(new)
            assert old["options"] == new["options"]
            assert old["target"] == new["target"]
            assert old["group"] == new["group"]
            if variant == "interpolation":
                for a, b in zip(json.loads(old["state"]), json.loads(new["state"]), strict=True):
                    assert b["cost"] == 7 * a["cost"] + 19
    assert base == original
    with pytest.raises(ValueError):
        transform(base[0], "unknown")


def test_overlap_and_fixed_selected_weights():
    rows = build(97, 4, "development")
    reject_overlap(rows, build(17, 4, "train"))
    with pytest.raises(ValueError, match="latent"):
        reject_overlap(rows, [transform(rows[0], "combined")])
    with pytest.raises(ValueError, match="content"):
        reject_overlap(rows, [dict(rows[0], group="other")])
    with pytest.raises(ValueError, match="duplicate"):
        reject_overlap(rows + rows, [])
    assert CHECKPOINTS["candidate"][1] == (
        "f287bb0138fd0f264331cd2fc983567bc68323a85a25cda467387fffae7bc93b"
    )
