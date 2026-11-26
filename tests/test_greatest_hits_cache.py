import copy

import numpy as np
import pytest

from scripts.greatest_hits_tasks import tasks_for_parent


def corpus():
    return {split: tasks_for_parent(parent, split, digest * 64,
            [dict(seconds=1., fields=['wood', 'hit', 'static']),
             dict(seconds=2., fields=[None, 'scratch', None])], set())[0]
            for split, parent, digest in [('train', '2015-01-01-00-00-01', 'a'),
                                          ('validation', '2015-01-02-00-00-01', 'b')]}


def test_all_frozen_questions_bind_once_to_native_events_without_target_inputs():
    from scripts.greatest_hits_cache import frozen_events

    splits = corpus()
    original = copy.deepcopy(splits)
    events = frozen_events(splits)
    assert len(events) == 4
    assert [x['split'] for x in events] == ['train']*2 + ['validation']*2
    assert [x['source_line'] for x in events] == [1, 2, 1, 2]
    assert [x['questions'] for x in events] == [3, 1, 3, 1]
    assert all('target' not in x and 'options' not in x and 'question' not in x for x in events)
    assert frozen_events({s: rows[::-1] for s, rows in splits.items()}) == events
    assert splits == original


@pytest.mark.parametrize('case', ['duplicate', 'date', 'asset', 'inconsistent', 'options', 'id'])
def test_corrupt_or_leaking_event_binding_fails(case):
    from scripts.greatest_hits_cache import frozen_events

    splits = corpus()
    if case == 'duplicate':
        splits['train'].append(copy.deepcopy(splits['train'][0]))
    elif case == 'date':
        splits['validation'] = tasks_for_parent('2015-01-01-00-00-02', 'validation', 'b'*64,
            [dict(seconds=1., fields=['wood', 'hit', 'static'])], set())[0]
    elif case == 'asset':
        for row in splits['validation']:
            row['asset_sha256'] = 'a'*64
    elif case == 'inconsistent':
        splits['train'][1]['event_seconds'] = 1.001
    elif case == 'options':
        splits['train'][0]['options'].reverse()
    else:
        splits['train'][0]['event_id'] += 'other'
    with pytest.raises(ValueError):
        frozen_events(splits)


def test_stream_windows_retain_overlapping_frames_and_absolute_pts():
    from scripts.greatest_hits_cache import frame_windows

    frames = [(i / 20, np.full((2, 3, 3), i, np.uint8)) for i in range(31)]
    result = list(frame_windows(iter(frames), [(.1, .6), (.35, .85), (.9, 1.4)]))
    for (rgb, pts), (start, end) in zip(result, [(.1, .6), (.35, .85), (.9, 1.4)], strict=True):
        expected = [i for i in range(31) if start <= i / 20 < end]
        np.testing.assert_array_equal(rgb[:, 0, 0, 0], expected)
        np.testing.assert_array_equal(pts, np.array(expected) / 20)


@pytest.mark.parametrize('case', ['order', 'nan', 'short', 'window'])
def test_invalid_stream_or_window_fails(case):
    from scripts.greatest_hits_cache import frame_windows

    frames = [(i / 20, np.zeros((2, 3, 3), np.uint8)) for i in range(21)]
    windows = [(.1, .6)]
    if case == 'order':
        frames[5] = (frames[4][0], frames[5][1])
    elif case == 'nan':
        frames[5] = (float('nan'), frames[5][1])
    elif case == 'short':
        frames = frames[:3]
    else:
        windows = [(.1, .7)]
    with pytest.raises(ValueError):
        list(frame_windows(iter(frames), windows))
