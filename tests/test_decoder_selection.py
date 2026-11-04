import pytest

from scripts.evaluate_decoder_transfer import require_selection


def fixture():
    h = [dict(step=i, selector=2 - i / 4096) for i in range(0, 2049, 256)]
    r = dict(status="PASS", steps=2048, best_step=2048, reload_exact=True)
    return r, h


def test_requires_completed_fixed_rule():
    r, h = fixture()
    assert require_selection(r, h) == 2048
    for changed in [dict(r, steps=256), dict(r, best_step=0), dict(r, reload_exact=False)]:
        with pytest.raises(ValueError):
            require_selection(changed, h)
    with pytest.raises(ValueError):
        require_selection(r, h[:-1])


def test_first_tied_checkpoint_and_finite():
    r, h = fixture()
    for x in h:
        x["selector"] = 1
    r["best_step"] = 0
    assert require_selection(r, h) == 0
    h[2]["selector"] = float("nan")
    with pytest.raises(ValueError):
        require_selection(r, h)
