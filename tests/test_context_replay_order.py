import pytest

from scripts.context_replay_order import replay_order


def test_fixed_equal_source_batches_preserve_context_order():
    a = [dict(id=f'a{i}') for i in range(12)]
    b = [dict(id=f'b{i}') for i in range(20)]
    rows = replay_order(a, b, 3)
    assert rows == replay_order(a, b, 3)
    assert len({r['id'] for r in rows}) == 24
    assert [r['id'] for r in rows if r['replay_source'] == 'context'] == [r['id'] for r in a]
    for i in range(0, 24, 8):
        assert [r['replay_source'] for r in rows[i:i+8]] == ['context']*4 + ['original']*4
    assert all('replay_source' not in r for r in a+b)


def test_insufficient_or_duplicate_data_rejected():
    with pytest.raises(ValueError):
        replay_order([], [], 1)
    with pytest.raises(ValueError):
        replay_order([dict(id='a')]*4, [dict(id=f'b{i}') for i in range(4)], 1)
