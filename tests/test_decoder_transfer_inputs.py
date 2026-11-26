import json

from scripts.build_conditional_curriculum import build, verify
from scripts.prepare_decoder_transfer import VARIANTS, transform


def test_transfer_keeps_targets_and_all_options():
    for row in build(179, 2, "development"):
        for variant in VARIANTS:
            changed = transform(row, variant)
            verify(changed)
            assert changed["target"] == row["target"]
            assert sorted(changed["options"]) == sorted(row["options"])
            if variant == "option_reverse":
                assert changed["options"] == row["options"][::-1]
            if variant == "state_reverse":
                assert json.loads(changed["state"]) == json.loads(row["state"])[::-1]


def test_identity_and_determinism():
    rows = build(179, 2, "development")
    a = [transform(r, v) for v in VARIANTS for r in rows]
    assert a == [transform(r, v) for v in VARIANTS for r in rows]
    assert len({r["id"] for r in a}) == len(a)
