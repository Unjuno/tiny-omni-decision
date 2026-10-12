import pytest

from tiny_omni_decision.evidence_decision import evidence_gated_json

OPTIONS = [{"option_id": "x", "class_id": "a"}, {"option_id": "y", "class_id": "b"}]


def test_missing_evidence_never_invokes_predictor():
    def predict():
        pytest.fail("inference must not run")
    result = evidence_gated_json(predict, ['a', 'b'], OPTIONS,
                                 required=['video', 'sensor'], available={'video': True})
    assert result['status'] == 'abstain'
    assert result['missing_modalities'] == ['sensor']
    assert all(o['probability'] is None for o in result['options'])


def test_valid_zero_signal_can_be_available():
    result = evidence_gated_json(lambda: [.8, .2], ['a', 'b'], OPTIONS,
                                 required=['sensor'], available={'sensor': True})
    assert result['selected_option_id'] == 'x'


@pytest.mark.parametrize('flag', [0, 1, None, 'true'])
def test_non_boolean_flags_rejected(flag):
    with pytest.raises(ValueError, match='boolean'):
        evidence_gated_json(lambda: [.5, .5], ['a', 'b'], OPTIONS,
                            required=['video'], available={'video': flag})


def test_missing_input_does_not_hide_invalid_options():
    with pytest.raises(ValueError):
        evidence_gated_json(lambda: [.5, .5], ['a', 'b'], OPTIONS[:1],
                            required=['video'], available={})


def test_alignment_gate_uses_exact_source_and_causal_index():
    from test_observation_alignment import batch

    from tiny_omni_decision.evidence_decision import aligned_evidence_json

    seen = []

    def predict(selected):
        seen.append(selected)
        return [.8, .2]

    result = aligned_evidence_json(predict, batch(), ['a', 'b'], [
                                       {'option_id': 'a', 'class_id': 'a'},
                                       {'option_id': 'b', 'class_id': 'b'}],
                                   required_sources={'video': 'v', 'audio': 'a'},
                                   query_time_ms=20, max_age_ms=12)
    assert result['status'] == 'selected'
    assert seen == [{'v': {'index': 2, 'age_ms': 0}, 'a': {'index': 1, 'age_ms': 5}}]
    stale = aligned_evidence_json(predict, batch(), ['a', 'b'], [
                                       {'option_id': 'a', 'class_id': 'a'},
                                       {'option_id': 'b', 'class_id': 'b'}],
                                  required_sources={'video': 'v', 'audio': 'a'},
                                  query_time_ms=100, max_age_ms=12)
    assert stale['status'] == 'abstain'
    assert len(seen) == 1


def test_alignment_gate_does_not_substitute_another_source():
    from test_observation_alignment import batch

    from tiny_omni_decision.evidence_decision import aligned_evidence_json

    result = aligned_evidence_json(lambda _: pytest.fail('unexpected inference'),
                                   batch(), ['a', 'b'], [
                                       {'option_id': 'a', 'class_id': 'a'},
                                       {'option_id': 'b', 'class_id': 'b'}],
                                   required_sources={'video': 'different_camera'},
                                   query_time_ms=20, max_age_ms=12)
    assert result['status'] == 'abstain'


def test_batch_option_mismatch_rejected_before_inference():
    from test_observation_alignment import batch

    from tiny_omni_decision.evidence_decision import aligned_evidence_json

    with pytest.raises(ValueError, match="batch IDs"):
        aligned_evidence_json(lambda _: pytest.fail("unexpected inference"),
                              batch(), ['a', 'b'], OPTIONS,
                              required_sources={'video': 'v'},
                              query_time_ms=20, max_age_ms=12)
