import copy

import numpy as np
import pytest

from scripts.greatest_hits_tasks import tasks_for_parent


def source_rows():
    return {s: tasks_for_parent(p, s, h*64,
        [dict(seconds=1., fields=['wood', 'hit', 'static'])], set())[0]
        for s, p, h in [('train', '2015-01-01-00-00-01', 'a'),
                        ('validation', '2015-01-02-00-00-01', 'b')]}


def test_query_inventory_has_all_options_and_no_target_dependence():
    from scripts.greatest_hits_audio_query import query_inventory

    rows = source_rows()
    before = copy.deepcopy(rows)
    result = query_inventory(rows)
    assert len(result) == 24
    assert all(set(r) == {'key', 'question', 'candidate'} for r in result)
    assert len({r['key'] for r in result}) == 24
    assert result == sorted(result, key=lambda x: x['key'])
    assert query_inventory({s: r[::-1] for s, r in rows.items()}) == result
    assert rows == before
    for split in rows.values():
        for row in split:
            row['target'] = (row['target']+1) % len(row['options'])
    assert query_inventory(rows) == result


def test_audio_positions_match_half_second_event_clock_without_fake_video_pts():
    from scripts.greatest_hits_audio_query import audio_positions

    positions = audio_positions()
    assert positions['speech'].shape == (25, 3)
    np.testing.assert_allclose(positions['speech'][:, 0], np.arange(25)*.04, atol=1e-7)
    assert not np.any(positions['speech'][:, 1:])
    np.testing.assert_array_equal(positions['event_audio'], [[.5, 0., 0.]])
    assert all(p.dtype == np.float32 for p in positions.values())


def test_bank_write_is_exact_once_and_other_events_stay_unset():
    from scripts.greatest_hits_audio_query import store_audio_event

    banks = {'speech': np.full((2,25,384), np.nan, np.float32),
             'event_audio': np.full((2,1,384), np.nan, np.float32)}
    values = {'speech': np.ones((25,384), np.float32),
              'event_audio': np.full((1,384), 2., np.float32)}
    store_audio_event(banks, 1, values)
    for kind in banks:
        np.testing.assert_array_equal(banks[kind][1], values[kind])
        assert np.isnan(banks[kind][0]).all()
    with pytest.raises(ValueError, match='already written'):
        store_audio_event(banks, 1, values)


@pytest.mark.parametrize('case', ['nan', 'dtype', 'shape', 'index', 'keys'])
def test_bad_audio_bank_write_fails_before_any_mutation(case):
    from scripts.greatest_hits_audio_query import store_audio_event

    banks = {'speech': np.full((2,25,384), np.nan, np.float32),
             'event_audio': np.full((2,1,384), np.nan, np.float32)}
    values = {'speech': np.ones((25,384), np.float32),
              'event_audio': np.ones((1,384), np.float32)}
    index = 0
    if case == 'nan':
        values['event_audio'][0,0] = np.nan
    elif case == 'dtype':
        values['event_audio'] = values['event_audio'].astype(np.float64)
    elif case == 'shape':
        values['speech'] = values['speech'][:24]
    elif case == 'index':
        index = -1
    else:
        del values['event_audio']
    with pytest.raises(ValueError):
        store_audio_event(banks, index, values)
    assert all(np.isnan(x).all() for x in banks.values())
