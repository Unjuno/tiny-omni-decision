import pytest

from scripts.build_grid_direct_speaker_pilot import (
    build_candidate_quartets,
    select_pilot,
)


def test_quartet_builder_is_deterministic_and_balances_all_options():
    codes = {'aaaaaa', 'bbaaaa', 'ccaaaa', 'ddaaaa'}

    forward, missing_forward = build_candidate_quartets('s13', codes)
    reverse, missing_reverse = build_candidate_quartets('s13', set(reversed(sorted(codes))))

    assert forward == reverse
    assert missing_forward == missing_reverse == 0
    assert len(forward) == 1
    row = forward[0]
    assert len(row['codes']) == len(row['option_ids']) == len(row['queries']) == 4
    assert {(q['audio_code'], q['target_id']) for q in row['queries']} == set(
        zip(row['codes'], row['option_ids'], strict=True)
    )
    assert all(sum(a != b for a, b in zip(row['anchor'], code, strict=True)) == 2
               for code in row['codes'] if code != row['anchor'])


def test_pilot_selection_is_per_speaker_hash_ordered_and_rejects_missing_speakers():
    rows = []
    for speaker, prefix in [('s13', 'aa'), ('s18', 'bb')]:
        codes = {prefix + 'aaaa', prefix + 'bbaa', prefix + 'ccaa', prefix + 'ddaa'}
        row, _ = build_candidate_quartets(speaker, codes)
        assert row
        rows.append(row[0])

    selected = select_pilot(rows, speakers=('s13', 's18'), per_speaker=1)
    assert [row['speaker'] for row in selected] == ['s13', 's18']
    assert select_pilot(list(reversed(rows)), speakers=('s13', 's18'), per_speaker=1) == selected


def test_quartet_builder_counts_anchors_without_three_neighbors():
    rows, missing = build_candidate_quartets('s13', {'aaaaaa', 'bbbbbb'})
    assert rows == []
    assert missing == 2


def test_quartet_builder_rejects_malformed_code_ids():
    with pytest.raises(ValueError, match='valid six-character codes'):
        build_candidate_quartets('s13', {'NOT-A-CODE'})
