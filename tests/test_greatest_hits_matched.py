import copy
from collections import Counter

import pytest


def data():
    joint = [dict(id=f'j{i}', split='train', task='material' if i < 2 else 'action',
                  target=i % 2, options=['x', 'y']) for i in range(7)]
    common = [dict(id=f'{m}{i}', split='train', modality=m, source='source',
                   cohort='cohort', task='small' if i == 0 else 'large',
                   target_index=i % 2, options=['x', 'y'])
              for m in ('text', 'image', 'audio', 'video') for i in range(5)]
    return joint, common


def test_deterministic_balanced_complete_cycles_before_repeats_and_exhaustion():
    from scripts.greatest_hits_matched import build_plan

    j, c = data()
    before = copy.deepcopy((j, c))
    plan = build_plan(j, c, steps=12)
    assert plan == build_plan(j[::-1], c[::-1], steps=12)
    assert (j, c) == before
    assert len(plan['order']) == 12 and all(len(b) == 8 for b in plan['order'])
    streams = {'joint': [i for b in plan['order'] for i in b[:4]]}
    streams.update({m: [b[k+4] for b in plan['order']]
                    for k, m in enumerate(('text', 'image', 'audio', 'video'))})
    for name, ids in streams.items():
        pool = {r['id'] for r in (j if name == 'joint' else c)
                if name == 'joint' or r['modality'] == name}
        for start in range(0, len(ids), len(pool)):
            part = ids[start:start+len(pool)]
            assert len(part) == len(set(part)) and set(part) <= pool
            if len(part) == len(pool):
                assert set(part) == pool
        assert plan['streams'][name]['consumed'] == len(ids)
        assert plan['streams'][name]['unique_questions'] == len(pool)
        assert plan['streams'][name]['exhausted_buckets']
    counts = Counter(streams['joint'][:7])
    assert set(counts.values()) == {1}
    j[0]['target'] = 1-j[0]['target']
    assert build_plan(j, c, steps=12) == plan  # Valid labels never affect sampling.


@pytest.mark.parametrize('case', ['validation', 'duplicate', 'cross_duplicate', 'missing_modality',
                                 'invalid_target', 'bool_target', 'duplicate_option', 'one_option'])
def test_invalid_corpus_fails_closed(case):
    from scripts.greatest_hits_matched import build_plan

    j, c = data()
    if case == 'validation':
        j[0]['split'] = 'validation'
    elif case == 'duplicate':
        j.append(copy.deepcopy(j[0]))
    elif case == 'cross_duplicate':
        c[0]['id'] = j[0]['id']
    elif case == 'missing_modality':
        c = [r for r in c if r['modality'] != 'audio']
    elif case == 'invalid_target':
        c[0]['target_index'] = 2
    elif case == 'bool_target':
        j[0]['target'] = True
    elif case == 'duplicate_option':
        j[0]['options'] = ['x', 'x']
    else:
        j[0]['options'] = ['x']
    with pytest.raises(ValueError):
        build_plan(j, c, steps=12)


@pytest.mark.parametrize('steps', [0, True, -1, 32769, 2.5])
def test_invalid_budget_rejected(steps):
    from scripts.greatest_hits_matched import build_plan

    with pytest.raises(ValueError):
        build_plan(*data(), steps=steps)


def test_full_candidate_permutation_and_original_target_alignment():
    from scripts.greatest_hits_matched import presentation

    j, c = data()
    for row in j+c:
        for occurrence in range(8):
            p = presentation(row, occurrence)
            assert p == presentation(row, occurrence)
            assert sorted(p['order']) == list(range(len(row['options'])))
            assert p['order'][p['target']] == row.get('target', row.get('target_index'))
    with pytest.raises(ValueError):
        presentation(j[0], True)


def retained():
    return dict(macro={'nll': 1.5}, modality_macro={
        m: {'accuracy': .6} for m in ('text', 'image', 'audio', 'video')})


def test_selection_requires_retention_then_joint_nll_earliest_tie():
    from scripts.greatest_hits_matched import select_checkpoint

    ref = retained()
    assert select_checkpoint(None, 2., ref, ref)['selected']
    assert not select_checkpoint(2., 2.-1e-6, ref, ref)['selected']
    assert select_checkpoint(2., 1.9, ref, ref)['selected']
    bad = retained()
    bad['modality_macro']['audio']['accuracy'] = .579
    rejected = select_checkpoint(2., .1, bad, ref)
    assert not rejected['eligible'] and not rejected['selected']
    assert 'audio_accuracy_drop_over_2pp' in rejected['reasons']
    bad = retained()
    bad['macro']['nll'] += .00002
    assert not select_checkpoint(2., .1, bad, ref)['eligible']


@pytest.mark.parametrize('case', ['nonfinite', 'negative', 'missing', 'invalid_accuracy'])
def test_invalid_selection_metrics_fail_closed(case):
    from scripts.greatest_hits_matched import select_checkpoint

    current = retained()
    score = 1.
    if case == 'nonfinite':
        score = float('nan')
    elif case == 'negative':
        score = -1.
    elif case == 'missing':
        del current['modality_macro']['video']
    else:
        current['modality_macro']['text']['accuracy'] = 1.1
    with pytest.raises(ValueError):
        select_checkpoint(2., score, current, retained())
