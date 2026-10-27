import hashlib

import pytest

from scripts.select_grid_av_pilot import select_pilot


def rows_for(speakers=('s1', 's2'), count=12):
    rows = []
    for speaker in speakers:
        for index in range(count):
            codes = [f'{speaker}-{index}-{slot}' for slot in range(4)]
            option_ids = [hashlib.sha256(code.encode()).hexdigest() for code in codes]
            rows.append({
                'speaker': speaker,
                'quartet_id': hashlib.sha256(f'{speaker}-{index}'.encode()).hexdigest(),
                'codes': codes,
                'option_ids': option_ids,
                'queries': [
                    {'audio_code': code, 'target_id': option_ids[position]}
                    for position, code in enumerate(codes)
                ],
            })
    return rows


def test_select_pilot_is_hash_ordered_balanced_and_repeatable():
    rows = rows_for()
    expected = ['s1', 's2']
    selected = select_pilot(rows, expected, per_speaker=3)
    assert selected == select_pilot(list(reversed(rows)), expected, per_speaker=3)
    assert [row['speaker'] for row in selected] == ['s1'] * 3 + ['s2'] * 3
    for speaker in expected:
        actual = [row['quartet_id'] for row in selected if row['speaker'] == speaker]
        wanted = sorted(
            (row['quartet_id'] for row in rows if row['speaker'] == speaker),
            key=lambda quartet_id: hashlib.sha256(
                f'grid-av-pilot-v1|{quartet_id}'.encode('ascii')
            ).hexdigest(),
        )[:3]
        assert actual == wanted


def test_select_pilot_rejects_missing_speaker_and_duplicate_quartet():
    with pytest.raises(ValueError, match='speaker'):
        select_pilot(rows_for(('s1',)), ['s1', 's2'], per_speaker=2)
    rows = rows_for(('s1',), count=3)
    rows[-1]['quartet_id'] = rows[0]['quartet_id']
    with pytest.raises(ValueError, match='duplicate'):
        select_pilot(rows, ['s1'], per_speaker=2)


def test_select_pilot_rejects_unbalanced_role_labels():
    rows = rows_for(('s1',), count=3)
    rows[0]['queries'][0]['target_id'] = rows[0]['option_ids'][1]
    with pytest.raises(ValueError, match='balanced'):
        select_pilot(rows, ['s1'], per_speaker=2)
