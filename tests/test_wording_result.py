import pytest

from scripts.verify_wording_result import verify_run


def fixture():
    config = dict(parent_ids=list(range(5120)) + list(range(3072)))
    result = dict(status="PASS_EXECUTION_ONLY", smoke=False, steps=1024,
                  reload_exact=True, best_step=0)
    history = [dict(step=i, selector=1+i, consumed=i*8, unique_parent_questions=5120,
                    source_counts=dict(original=4096, curriculum=4096))
               for i in range(0, 1025, 128)]
    return config, result, history


def test_completed_run_and_fail_closed_partial():
    c, r, h = fixture()
    verify_run(c, r, h)
    with pytest.raises(ValueError, match="evaluation"):
        verify_run(c, r, h[:-1])
    r["reload_exact"] = False
    with pytest.raises(ValueError, match="reload"):
        verify_run(c, r, h)


def test_last_checkpoint_is_not_automatically_best():
    c, r, h = fixture()
    r["best_step"] = 1024
    with pytest.raises(ValueError, match="selected"):
        verify_run(c, r, h)
