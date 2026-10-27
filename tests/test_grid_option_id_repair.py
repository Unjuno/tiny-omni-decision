import pytest

from scripts.repair_grid_option_ids import repair


def test_repair_preserves_code_order_and_balances_local_ids():
    codes = ['a', 'b', 'c', 'd']
    ids = ['hash-a', 'hash-b', 'hash-c', 'hash-d']
    row = {'codes': codes, 'option_ids': ids,
           'queries': [{'audio_code': c, 'target_id': i}
                       for c, i in zip(codes, ids, strict=True)]}
    result = repair(row)
    assert result['codes'] == codes
    assert result['option_ids'] == [f'option-{i}' for i in range(4)]
    assert {q['target_id'] for q in result['queries']} == set(result['option_ids'])
    assert row['option_ids'] == ids


def test_repair_rejects_mismatched_target():
    row = {'codes': ['a', 'b', 'c', 'd'], 'option_ids': ['1', '2', '3', '4'],
           'queries': [{'audio_code': c, 'target_id': '1'} for c in 'abcd']}
    with pytest.raises(ValueError, match='target mapping'):
        repair(row)
