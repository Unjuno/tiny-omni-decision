import json

import pytest

from scripts.build_context_curriculum import context_rows


def test_deterministic_grouped_context_preserves_targets():
    rows = context_rows(211, 4, "train")
    assert rows == context_rows(211, 4, "train")
    assert len(rows) == 72 and len({r['id'] for r in rows}) == 72
    for offset in range(0, len(rows), 3):
        group = rows[offset:offset + 3]
        for key in ['group', 'state', 'options', 'target', 'oracle']:
            assert all(r[key] == group[0][key] for r in group)
        necessary = group[2]
        assert necessary['oracle']['field'] in necessary['question'].split('.')[0]
        assert necessary['oracle']['field'] not in necessary['question'].split('.')[1]
        state = json.loads(necessary['state'])
        f = necessary['oracle']['field']
        chosen = sorted(state, key=lambda item: item[f])
        assert necessary['target'] == chosen[
            -1 if necessary['oracle']['direction'] == 'maximum' else 0
        ]['id']


def test_separate_states_and_heldout_context_templates():
    train = context_rows(211, 4, 'train')
    dev = context_rows(223, 4, 'development')
    assert not {r['group'] for r in train} & {r['group'] for r in dev}
    for kind in ['necessary', 'irrelevant']:
        assert not {r['question'] for r in train if r['context'] == kind} & {
            r['question'] for r in dev if r['context'] == kind
        }


def test_unknown_split_rejected():
    with pytest.raises(ValueError):
        context_rows(17, 1, 'audit')


def test_training_arms_share_order_and_balance_context():
    from collections import Counter

    from scripts.train_context_robustness import ordered_views

    rows = context_rows(211, 512, 'train')
    arms = ordered_views(rows)
    assert arms == ordered_views(rows)
    assert [r['base_id'] for r in arms['plain']] == [r['base_id'] for r in arms['context']]
    assert len({r['base_id'] for r in arms['context']}) == 3072
    assert Counter(r['context'] for r in arms['context']) == {
        'plain': 1024, 'irrelevant': 1024, 'necessary': 1024
    }
