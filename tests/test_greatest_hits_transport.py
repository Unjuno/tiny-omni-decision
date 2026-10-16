import copy

import numpy as np
import pytest

from scripts.greatest_hits_audio_query import query_inventory
from scripts.greatest_hits_cache import frozen_events
from scripts.greatest_hits_tasks import tasks_for_parent


def fixture():
    splits = {s: tasks_for_parent(p, s, h*64,
        [dict(seconds=1., fields=['wood', 'hit', 'static'])], set())[0]
        for s, p, h in [('train', '2015-01-01-00-00-01', 'a'),
                        ('validation', '2015-01-02-00-00-01', 'b')]}
    events = frozen_events(splits)
    parents = [dict(parent=e['parent'], split=e['split'], events=[dict(**e, bank_index=0,
        timing=dict(start_seconds=.75, end_seconds=1.25, first_sample=72000,
                    end_sample_exclusive=120000, center_error_seconds=0.))]) for e in events]
    common = dict(phase='full', completed_events=2, completed_parents=2,
                  completed_questions=6, expected_events=2, expected_parents=2,
                  expected_questions=6, parents=parents, artifacts={})
    audio = dict(**copy.deepcopy(common), status='PASS_COMPLETE_AUDIO_QUERY_CACHE')
    video = dict(**copy.deepcopy(common), status='PASS_COMPLETE_VIDEO_CACHE')
    return splits, audio, video, dict(events=events, queries=query_inventory(splits)), events


def test_bind_complete_cache_without_targets_and_preserve_original_option_order():
    from scripts.greatest_hits_transport import bind_caches

    args = fixture()
    before = copy.deepcopy(args)
    result = bind_caches(*args, phase='full')
    assert len(result['events']) == 2 and len(result['questions']) == 6
    original = args[0]['train'][0]
    assert result['questions'][original['id']]['options'] == original['options']
    assert 'target' not in repr(result)
    assert args == before
    for rows in args[0].values():
        for row in rows:
            row['target'] = (row['target']+1) % len(row['options'])
    assert bind_caches(*args, phase='full') == result


@pytest.mark.parametrize('case', ['running', 'missing', 'duplicate', 'clock', 'asset',
                                 'index', 'split', 'plan', 'query', 'count'])
def test_invalid_cache_binding_rejected(case):
    from scripts.greatest_hits_transport import bind_caches

    args = list(fixture())
    if case == 'running':
        args[1]['status'] = 'RUNNING'
    elif case == 'missing':
        args[2]['parents'].pop()
    elif case == 'duplicate':
        args[2]['parents'].append(copy.deepcopy(args[2]['parents'][0]))
    elif case == 'clock':
        args[2]['parents'][0]['events'][0]['timing']['start_seconds'] = .8
    elif case == 'asset':
        args[2]['parents'][0]['events'][0]['asset_sha256'] = 'c'*64
    elif case == 'index':
        args[1]['parents'][0]['events'][0]['bank_index'] = 1
    elif case == 'split':
        args[2]['parents'][0]['events'][0]['split'] = 'validation'
    elif case == 'plan':
        args[4].reverse()
    elif case == 'query':
        args[3]['queries'][0]['candidate'] = 'wrong content'
    else:
        args[1]['completed_questions'] = 5
    with pytest.raises(ValueError):
        bind_caches(*args, phase='full')


def test_smoke_rejects_validation_media_and_full_rejects_partial_subset():
    from scripts.greatest_hits_transport import bind_caches

    args = list(fixture())
    args[1]['phase'] = args[2]['phase'] = 'smoke'
    with pytest.raises(ValueError):
        bind_caches(*args, phase='smoke')
    with pytest.raises(ValueError):
        bind_caches(*args, phase='full')


def test_explicit_positions_are_exact_and_bad_geometry_rejected():
    from scripts.greatest_hits_features import event_positions
    from scripts.greatest_hits_transport import validate_positions

    pts = np.arange(8, dtype=np.float64)*.06+.76
    positions = event_positions(pts, .75, 1.25)
    validate_positions(positions, pts, fixture()[1]['parents'][0]['events'][0]['timing'])
    broken = copy.deepcopy(positions)
    broken['video'][:, 0] = np.repeat(np.linspace(0, 1, 8), 256)
    with pytest.raises(ValueError):
        validate_positions(broken, pts, fixture()[1]['parents'][0]['events'][0]['timing'])
    broken = copy.deepcopy(positions)
    broken['speech'][:, 0] = np.linspace(0, 1, 25)
    with pytest.raises(ValueError):
        validate_positions(broken, pts, fixture()[1]['parents'][0]['events'][0]['timing'])


def store_fixture(tmp_path):
    from scripts.greatest_hits_audio_query import audio_positions
    from scripts.greatest_hits_features import event_positions
    from scripts.greatest_hits_transport import bind_caches
    from scripts.native_video_cache import sha

    args = fixture()
    binding = bind_caches(*args, phase='full')
    dirs = {k: tmp_path/k for k in ('audio', 'video')}
    hashes = {k: {} for k in dirs}
    for path in dirs.values():
        path.mkdir()
    def save(kind, name, value):
        np.save(dirs[kind]/name, value, allow_pickle=False)
        hashes[kind][name] = sha(dirs[kind]/name)
    for kind, values in audio_positions().items():
        save('audio', kind+'-positions.npy', values)
    save('audio', 'queries.npy', np.arange(24*384, dtype=np.float32).reshape(24,384)/1000)
    pts = np.arange(8, dtype=np.float64)*.06+.76
    for event in args[4]:
        parent = event['parent']
        save('audio', parent+'-speech.npy', np.full((1,25,384), .1, np.float32))
        save('audio', parent+'-event_audio.npy', np.full((1,1,384), .2, np.float32))
        save('video', parent+'-video.npy', np.full((1,2048,768), .3, np.float32))
        save('video', parent+'-positions.npy', event_positions(pts,.75,1.25)['video'][None])
        save('video', parent+'-pts.npy', pts[None])
    return binding, dirs, hashes


def test_store_uses_exact_positions_permutations_and_explicit_missing_views(tmp_path):
    torch = pytest.importorskip('torch')
    from scripts.greatest_hits_transport_store import ExplicitEventStore

    binding, dirs, hashes = store_fixture(tmp_path)
    store = ExplicitEventStore(binding, dirs, hashes, device='cpu')
    ident = next(iter(binding['questions']))
    blocks, queries = store.get(ident)
    assert [b.kind for b in blocks] == ['speech', 'event_audio', 'video']
    torch.testing.assert_close(blocks[0].positions[:,0], torch.arange(25)*.04,
                               atol=1e-7, rtol=0)
    torch.testing.assert_close(blocks[2].positions[::256,0],
                               torch.arange(8)*.12+.02, atol=1e-7, rtol=0)
    _, permuted = store.get(ident, order=list(reversed(range(len(queries)))))
    torch.testing.assert_close(permuted, queries.flip(0), atol=0, rtol=0)
    assert [b.kind for b in store.get(ident, mode='audio')[0]] == ['speech', 'event_audio']
    assert [b.kind for b in store.get(ident, mode='video')[0]] == ['video']
    assert all(b.valid.all() for b in blocks)
    with pytest.raises(ValueError):
        store.get(ident, order=[0,0])
    with pytest.raises(ValueError):
        store.get(ident, mode='missing')
    with pytest.raises(ValueError):
        store.get('unregistered event')
    store.verify_files()


@pytest.mark.parametrize('case', ['hash', 'shape', 'finite', 'changed'])
def test_store_rejects_bad_and_changed_payloads(tmp_path, case):
    pytest.importorskip('torch')
    from scripts.greatest_hits_transport_store import ExplicitEventStore
    from scripts.native_video_cache import sha

    binding, dirs, hashes = store_fixture(tmp_path)
    ident = next(iter(binding['questions']))
    event = binding['events'][binding['questions'][ident]['event_id']]
    name = event['parent']+'-speech.npy'
    if case == 'changed':
        store = ExplicitEventStore(binding, dirs, hashes, device='cpu')
        store.get(ident)
        store.arrays.clear()
    bad = np.ones((1,25,384), np.float32)
    if case == 'shape':
        bad = bad[:,:24]
    elif case == 'finite':
        bad[0,0,0] = np.nan
    np.save(dirs['audio']/name, bad)
    if case in ('shape', 'finite'):
        hashes['audio'][name] = sha(dirs['audio']/name)
    if case != 'changed':
        store = ExplicitEventStore(binding, dirs, hashes, device='cpu')
    with pytest.raises(ValueError):
        store.get(ident)
