import pytest


def module():
    from scripts import native_language_cache
    return native_language_cache


def rows():
    result = {}
    for index, name in enumerate(('binding_train', 'numeric_train', 'legacy_train',
                                  'validation', 'old_validation', 'legacy_validation')):
        result[name] = [dict(id=str(index), group=str(index), state=f'state {index}',
                            question='which?', options=['yes', 'no'],
                            split='train' if index < 3 else 'validation')]
    return result


def test_inventory_preserves_order_and_queries_share_no_labels():
    data = rows()
    data['binding_train'].append({**data['binding_train'][0], 'id': 'extra',
                                  'options': ['no', 'yes']})
    inventory = module().inventory(data)
    assert len(inventory['states']) == 6
    assert len(inventory['pairs']) == 2
    mapped = inventory['rows']['binding_train']
    assert [r['id'] for r in mapped] == ['0', 'extra']
    assert mapped[0]['state'] == mapped[1]['state']
    assert mapped[0]['queries'] == list(reversed(mapped[1]['queries']))


def test_extra_reserved_split_rejected():
    data = rows()
    data['sealed_audit'] = []
    with pytest.raises(ValueError):
        module().inventory(data)


@pytest.mark.parametrize('field', ['state', 'group'])
def test_cross_split_observation_and_group_rejected(field):
    data = rows()
    data['validation'][0][field] = data['binding_train'][0][field]
    with pytest.raises(ValueError):
        module().inventory(data)


def test_wrong_declared_split_rejected():
    data = rows()
    data['validation'][0]['split'] = 'train'
    with pytest.raises(ValueError):
        module().inventory(data)


def test_raw_content_is_not_normalized_in_cache_keys():
    assert module().text_key('a  b') != module().text_key('a b')
    assert module().pair_key('a', 'bc') != module().pair_key('ab', 'c')


def test_duplicate_example_ids_rejected():
    data = rows()
    data['binding_train'].append(data['binding_train'][0])
    with pytest.raises(ValueError):
        module().inventory(data)


class LengthTokenizer:
    def __call__(self, text, text_pair=None, **kwargs):
        count = len(text.split()) + 2
        if text_pair is not None:
            count += len(text_pair.split()) + 1
        return {'input_ids': list(range(count))}


def test_storage_plan_separates_train_validation_and_preserves_all_rows():
    plan = module().storage_plan(module().inventory(rows()), LengthTokenizer())
    assert plan['shapes'] == {'train-states.npy': [12, 384],
                              'validation-states.npy': [12, 384], 'queries.npy': [2, 384]}
    assert plan['tensor_bytes'] == 26 * 384 * 4
    assert len(plan['rows']) == 6
    assert sorted(x['offset'] for x in plan['states'].values()) == [0, 0, 4, 4, 8, 8]


def test_oversized_query_fails_before_storage():
    data = rows()
    data['binding_train'][0]['question'] = 'word ' * 510
    with pytest.raises(ValueError):
        module().storage_plan(module().inventory(data), LengthTokenizer())


def test_oversized_state_fails_before_storage():
    data = rows()
    data['binding_train'][0]['state'] = 'word ' * 8191
    with pytest.raises(ValueError):
        module().storage_plan(module().inventory(data), LengthTokenizer())


def test_profile_cannot_claim_a_different_chunking_or_precision():
    config = dict(window_content_tokens=510, overlap_content_tokens=32, batch_size=1,
                  dtype='float32', attention='eager', seed=17)
    module().validate_profile(config)
    for field, changed in [('overlap_content_tokens', 16), ('dtype', 'bfloat16'),
                           ('batch_size', 8), ('window_content_tokens', 512)]:
        with pytest.raises(ValueError):
            module().validate_profile({**config, field: changed})
