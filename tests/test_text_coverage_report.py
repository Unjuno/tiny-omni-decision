import pytest

from scripts.report_text_coverage import verify_finished


def test_rejects_partial_and_incorrect_selection():
    history = [dict(step=i, selector=1 - i / 10000) for i in range(0, 2049, 128)]
    r = dict(
        status="PASS_EXECUTION_ONLY", smoke=False, steps=2048, best_step=2048, reload_exact=True
    )
    verify_finished(r, history)
    for altered in [
        dict(r, steps=128),
        dict(r, best_step=0),
        dict(r, reload_exact=False),
        dict(r, smoke=True),
    ]:
        with pytest.raises(ValueError):
            verify_finished(altered, history)
    with pytest.raises(ValueError):
        verify_finished(r, history[:-1])
