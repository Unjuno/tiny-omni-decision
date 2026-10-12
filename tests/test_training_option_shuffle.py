import copy
import random

from scripts.shuffle_training_options import shuffled_options


def test_only_options_change_without_global_rng_side_effects():
    rows = [dict(id=str(i), state="state", question="question", options=["a", "b", "c"],
                 target="b") for i in range(24)]
    before = copy.deepcopy(rows)
    rng = random.getstate()
    changed = shuffled_options(rows)
    assert random.getstate() == rng
    assert rows == before
    assert changed == shuffled_options(rows)
    assert any(a["options"] != b["options"] for a,b in zip(rows,changed,strict=True))
    for a,b in zip(rows,changed,strict=True):
        assert sorted(a["options"]) == sorted(b["options"])
        assert {k:v for k,v in a.items() if k != "options"} == {
            k:v for k,v in b.items() if k != "options"}
