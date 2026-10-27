import pytest

from scripts.prepare_grid_hard_correspondence import construct


def test_hard_quartet_has_four_balanced_roles():
    rows = [{'speaker': 's1', 'code': c} for c in ('bbaf2n', 'bbaf3n', 'bbaf4n', 'bbaf5n')]
    quartets, skipped = construct(rows)
    assert skipped == 0 and len(quartets) == 1
    q = quartets[0]
    assert {x['target_id'] for x in q['queries']} == set(q['option_ids'])
    assert {x['audio_code'] for x in q['queries']} == set(q['codes'])


def test_duplicate_source_code_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        construct([{'speaker': 's1', 'code': 'bbaf2n'}] * 2)
