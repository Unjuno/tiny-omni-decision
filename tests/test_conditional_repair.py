import pytest

from scripts.train_conditional_repair import schedule, verify_splits


def test_equal_unique_reproducible_schedule():
    sources = [list(range(3000)), list(range(8192))]
    batches = schedule(sources, 512)
    assert batches == schedule(sources, 512)
    flat = [pair for batch in batches for pair in batch]
    assert len(flat) == len(set(flat)) == 4096
    assert all(sum(s == 0 for s, _ in batch) == 4 for batch in batches)
    with pytest.raises(ValueError):
        schedule([list(range(3)), list(range(20))], 1)


def test_cross_source_state_overlap_and_duplicate_rejected():
    train = [{"id": "a", "state": '{"x":1,"y":2}'}]
    with pytest.raises(ValueError):
        verify_splits(train, [{"id": "b", "state": '{"y": 2, "x":1}'}])
    with pytest.raises(ValueError):
        verify_splits(train * 2, [{"id": "b", "state": "other"}])
    verify_splits(train, [{"id": "b", "state": "other"}])
