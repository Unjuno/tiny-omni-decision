import pytest

from tiny_omni_decision.evidence_decision import evidence_gated_json

OPTIONS = [{"option_id": "x", "class_id": "a"}, {"option_id": "y", "class_id": "b"}]


@pytest.mark.parametrize("times, valid, max_age, expected_index", [
    ((21,), (True,), 100, None),  # Future evidence cannot satisfy the condition.
    ((20,), (False,), 100, None),
    ((15, 21), (True, True), 5, 0),  # Inclusive boundary; never choose future.
    ((15,), (True,), 4.999, None),
    ((10, 20), (True, False), 10, 0),  # Latest valid evidence policy.
    ((20,), (True,), 0, 0),
])
def test_spoken_criterion_causal_boundaries(times, valid, max_age, expected_index):
    from test_observation_alignment import batch

    from tiny_omni_decision.evidence_decision import aligned_evidence_json
    from tiny_omni_decision.observations import Criterion

    request = batch()
    audio = request.observations[1].model_copy(update={
        "timestamps_ms": times, "valid_mask": valid,
    })
    request = request.model_copy(update={
        "observations": (request.observations[0], audio),
        "criterion": Criterion(kind="spoken", value="a"),
    })
    seen = []

    def predict(selected):
        seen.append(selected)
        return [.8, .2]

    result = aligned_evidence_json(
        predict, request, ['a', 'b'],
        [{'option_id': 'a', 'class_id': 'a'}, {'option_id': 'b', 'class_id': 'b'}],
        required_sources={'video': 'v'}, query_time_ms=20, max_age_ms=max_age,
    )
    if expected_index is None:
        assert result['status'] == 'abstain'
        assert seen == []
        assert result['selected_option_id'] is None
        assert all(o['probability'] is None for o in result['options'])
    else:
        assert result['status'] == 'selected'
        assert len(seen) == 1
        assert seen[0]['a'] == {'index': expected_index,
                                'age_ms': 20 - times[expected_index]}


@pytest.mark.parametrize("max_age, expected", [(4, "abstain"), (12, "selected")])
def test_spoken_criterion_requires_fresh_audio_even_for_visual_task(max_age, expected):
    from test_observation_alignment import batch

    from tiny_omni_decision.evidence_decision import aligned_evidence_json
    from tiny_omni_decision.observations import Criterion

    request = batch().model_copy(update={
        "criterion": Criterion(kind="spoken", value="a"),
    })
    seen = []

    def predict(selected):
        seen.append(selected)
        return [.8, .2]

    result = aligned_evidence_json(
        predict, request, ['a', 'b'],
        [{'option_id': 'a', 'class_id': 'a'}, {'option_id': 'b', 'class_id': 'b'}],
        required_sources={'video': 'v'}, query_time_ms=20, max_age_ms=max_age,
    )
    assert result['status'] == expected
    if expected == 'abstain':
        assert seen == []
        assert result['missing_modalities'] == ['audio']
    else:
        assert seen == [{'v': {'index': 2, 'age_ms': 0},
                         'a': {'index': 1, 'age_ms': 5}}]


def test_missing_evidence_never_invokes_predictor():
    def predict():
        pytest.fail("inference must not run")
    result = evidence_gated_json(predict, ['a', 'b'], OPTIONS,
                                 required=['video', 'sensor'], available={'video': True})
    assert result['status'] == 'abstain'
    assert result['missing_modalities'] == ['sensor']
    assert all(o['probability'] is None for o in result['options'])


@pytest.mark.parametrize("max_age", [4, 12])
def test_spoken_condition_does_not_replace_required_environment_audio(max_age):
    from test_observation_alignment import batch

    from tiny_omni_decision.evidence_decision import aligned_evidence_json
    from tiny_omni_decision.observations import Criterion

    request = batch()
    spoken = request.observations[1].model_copy(update={
        "source_id": "spoken", "timestamps_ms": (20,), "valid_mask": (True,),
    })
    request = request.model_copy(update={
        "observations": (*request.observations, spoken),
        "criterion": Criterion(kind="spoken", value="spoken"),
    })
    seen = []

    def predict(selected):
        seen.append(selected)
        return [.8, .2]

    result = aligned_evidence_json(
        predict, request, ['a', 'b'],
        [{'option_id': 'a', 'class_id': 'a'}, {'option_id': 'b', 'class_id': 'b'}],
        required_sources={'audio': 'a'}, query_time_ms=20, max_age_ms=max_age,
    )
    if max_age == 4:
        assert result['status'] == 'abstain'
        assert seen == []
    else:
        assert result['status'] == 'selected'
        assert set(seen[0]) == {'a', 'spoken'}


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
