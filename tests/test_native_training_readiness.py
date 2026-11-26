import pytest


def module():
    from scripts import native_training_readiness
    return native_training_readiness


def test_state_key_preserves_content_and_normalizes_json_order():
    key = module().state_key
    assert key('{"x":1,"y":2}') == key('{"y":2, "x":1}')
    assert key('plain observation') == key('plain   observation')
    assert key('plain observation') != key('other observation')
    assert key('{"x":1}') != key('{"x":2}')


def test_attribute_signature_is_separate_from_media_identity():
    key = module().state_key
    image = {'color': 'red', 'shape': 'cube', 'texture': 'metal', 'count': 2}
    text = '{"shape":"cube","color":"red","texture":"metal","count":"2"}'
    assert key(image) == key(text)
    result = module().overlaps({'train': {key(image): {'image'}},
                                'validation': {key(text): {'image'}}})
    assert result['shared_attribute_or_state_signatures'] == 1
    assert result['does_not_by_itself_establish_media_leakage']


def test_overlap_reports_direction_without_dropping_states():
    result = module().overlaps({'train': {'a': {'text'}, 'b': {'image'}},
                                'validation': {'a': {'image'}, 'b': {'image'}, 'c': {'text'}}})
    assert result['shared_attribute_or_state_signatures'] == 2
    assert result['source_pairs'] == {'image -> image': 1, 'text -> image': 1}


def test_token_length_measurement_never_truncates_or_filters():
    calls = []

    def tokenizer(texts, **kwargs):
        calls.append(kwargs)
        return {'input_ids': [[0] * int(t) for t in texts]}

    result = module().token_lengths(tokenizer, ['38', '513', '824'])
    assert result == {'count': 3, 'max_tokens': 824, 'over_512': 2, 'over_8192': 0}
    assert all(c['truncation'] is False for c in calls)


def test_empty_length_inventory_rejected():
    with pytest.raises(ValueError):
        module().token_lengths(lambda _: None, [])
