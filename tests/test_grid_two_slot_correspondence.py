from scripts.prepare_grid_two_slot_correspondence import construct


def test_two_slot_quartets_exclude_one_slot_neighbors_and_balance_roles():
    codes = ['aaaaaa', 'bbaaaa', 'ccaaaa', 'ddaaaa', 'baaaaa']
    rows = [{'speaker': 's1', 'code': code} for code in codes]
    quartets, _ = construct(rows)
    assert quartets
    for quartet in quartets:
        anchor = quartet['anchor']
        assert all(sum(a != b for a, b in zip(anchor, code, strict=True)) == 2
                   for code in quartet['codes'] if code != anchor)
        assert {q['target_id'] for q in quartet['queries']} == set(quartet['option_ids'])
